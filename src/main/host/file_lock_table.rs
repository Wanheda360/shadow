//! POSIX advisory file lock table for `fcntl(F_SETLK/F_SETLKW/F_GETLK)`.
//!
//! # Semantics modelled
//!
//! This implements **process-associated** POSIX locks (not OFD locks).  The
//! key rules are:
//!
//! * A process never conflicts with its own locks.  A new `F_SETLK` from the
//!   same PID over an overlapping range *replaces* the existing lock (or
//!   splits/merges as required).
//! * Two `F_RDLCK` (shared) locks are always compatible.
//! * An `F_WRLCK` (exclusive) lock conflicts with any other lock from a
//!   *different* process.
//! * `F_UNLCK` releases the requested range and wakes any waiters whose
//!   conflict is now gone.
//! * On `close()` or process exit the caller must call [`FileLockTable::release_all`]
//!   to drop every lock held by that process on the file.
//!
//! # File identity
//!
//! Locks are keyed by a [`FileKey`] — a stable `u64` that uniquely identifies
//! an *open-file description*.
//!
//! * For **`CompatFile::New`** files, use the raw address of the inner
//!   `Arc`/`RootedRc` allocation (`Arc::as_ptr(d.inner_file()) as u64`).
//!   Duplicated file descriptors (`dup`, `fork`) share the same Arc, so they
//!   share the same key — which is correct.
//!
//! * For **`CompatFile::Legacy`** files, the caller should derive the key from
//!   `(dev, ino)` obtained by calling `fstat` on the host fd exposed by the C
//!   layer (see TODO in `fcntl.rs`).  Until that plumbing exists a pointer to
//!   the `LegacyFile` object can be used as a fallback; it is correct for
//!   `fork()`-inherited descriptors but **not** for two independent `open()`
//!   calls on the same filesystem path.

use std::collections::HashMap;

use crate::host::process::ProcessId;
use crate::host::thread::ThreadId;

// ──────────────────────────────────────────────────────────────────────────────
// Public types
// ──────────────────────────────────────────────────────────────────────────────

/// Stable identifier for an open-file description (see module-level docs).
pub type FileKey = u64;

/// The type of a POSIX advisory lock.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LockType {
    /// `F_RDLCK` — shared read lock.
    Shared,
    /// `F_WRLCK` — exclusive write lock.
    Exclusive,
}

/// A thread that returned `SyscallError::Blocked` waiting to acquire a lock.
/// Stored in [`FileLockTable`]; the unlock path schedules `host.resume` for
/// each waiter that is no longer conflicted.
pub struct LockWaiter {
    pub pid: ProcessId,
    pub tid: ThreadId,
    pub start: u64,
    pub end: u64,
    pub lock_type: LockType,
}

// ──────────────────────────────────────────────────────────────────────────────
// Internal types
// ──────────────────────────────────────────────────────────────────────────────

/// A single acquired lock segment.
#[derive(Clone, Debug)]
struct Segment {
    /// Inclusive start byte of the locked range.
    start: u64,
    /// Exclusive end byte.  `u64::MAX` encodes "to end of file".
    end: u64,
    owner: ProcessId,
    lock_type: LockType,
}

impl Segment {
    fn overlaps(&self, start: u64, end: u64) -> bool {
        self.start < end && self.end > start
    }
}

/// Per-file lock state: acquired segments + blocked waiters.
#[derive(Default)]
struct FileLockState {
    segments: Vec<Segment>,
    waiters: Vec<LockWaiter>,
}

impl FileLockState {
    // ── Conflict detection ────────────────────────────────────────────────

    /// Returns the first segment that conflicts with a prospective lock
    /// `[start, end)` of `lock_type` by `requester`, or `None`.
    fn find_conflict(
        &self,
        start: u64,
        end: u64,
        lock_type: LockType,
        requester: ProcessId,
    ) -> Option<&Segment> {
        self.segments.iter().find(|seg| {
            // POSIX: same process never conflicts with itself.
            if seg.owner == requester {
                return false;
            }
            if !seg.overlaps(start, end) {
                return false;
            }
            // Two shared locks are always compatible.
            !matches!(
                (lock_type, seg.lock_type),
                (LockType::Shared, LockType::Shared)
            )
        })
    }

    // ── Acquisition ───────────────────────────────────────────────────────

    /// Acquire `[start, end)` with `lock_type` for `owner`.
    ///
    /// Caller **must** have verified that `find_conflict` returns `None` first.
    /// Existing segments owned by the same process that overlap the new range
    /// are split / replaced / merged so the invariant "no two same-owner
    /// segments overlap" is maintained.
    fn acquire(&mut self, start: u64, end: u64, lock_type: LockType, owner: ProcessId) {
        // 1. Remove overlapping same-owner segments, keeping non-overlapping
        //    tails as split remnants.
        let mut remnants: Vec<Segment> = Vec::new();
        self.segments.retain(|seg| {
            if seg.owner != owner || !seg.overlaps(start, end) {
                return true;
            }
            if seg.start < start {
                remnants.push(Segment {
                    start: seg.start,
                    end: start,
                    owner,
                    lock_type: seg.lock_type,
                });
            }
            if seg.end > end {
                remnants.push(Segment {
                    start: end,
                    end: seg.end,
                    owner,
                    lock_type: seg.lock_type,
                });
            }
            false
        });
        self.segments.extend(remnants);

        // 2. Push the new segment.
        self.segments.push(Segment {
            start,
            end,
            owner,
            lock_type,
        });

        // 3. Merge adjacent / overlapping same-owner+type segments.
        self.coalesce(owner, lock_type);
    }

    /// Merge adjacent or overlapping segments with the given `owner` and
    /// `lock_type`.  O(n²) but n is tiny in practice (BoltDB holds at most
    /// one whole-file lock per process).
    fn coalesce(&mut self, owner: ProcessId, lock_type: LockType) {
        loop {
            let mut merged = false;
            let n = self.segments.len();
            'outer: for i in 0..n {
                for j in (i + 1)..n {
                    let matches_criteria = {
                        let a = &self.segments[i];
                        let b = &self.segments[j];
                        a.owner == owner
                            && b.owner == owner
                            && a.lock_type == lock_type
                            && b.lock_type == lock_type
                            // adjacent or overlapping
                            && a.end >= b.start
                            && b.end >= a.start
                    };
                    if matches_criteria {
                        let new_start = self.segments[i].start.min(self.segments[j].start);
                        let new_end = self.segments[i].end.max(self.segments[j].end);
                        self.segments.remove(j); // higher index first
                        self.segments[i] = Segment {
                            start: new_start,
                            end: new_end,
                            owner,
                            lock_type,
                        };
                        merged = true;
                        break 'outer;
                    }
                }
            }
            if !merged {
                break;
            }
        }
    }

    // ── Release ───────────────────────────────────────────────────────────

    /// Release `[start, end)` owned by `owner`.
    ///
    /// Returns waiters that are no longer conflicted after the release.
    fn release(&mut self, start: u64, end: u64, owner: ProcessId) -> Vec<LockWaiter> {
        let mut remnants: Vec<Segment> = Vec::new();
        self.segments.retain(|seg| {
            if seg.owner != owner || !seg.overlaps(start, end) {
                return true;
            }
            if seg.start < start {
                remnants.push(Segment {
                    start: seg.start,
                    end: start,
                    owner,
                    lock_type: seg.lock_type,
                });
            }
            if seg.end > end {
                remnants.push(Segment {
                    start: end,
                    end: seg.end,
                    owner,
                    lock_type: seg.lock_type,
                });
            }
            false
        });
        self.segments.extend(remnants);
        self.drain_ready_waiters()
    }

    /// Remove and return any waiters that are no longer blocked by the current
    /// segment state.  Called after every release.
    fn drain_ready_waiters(&mut self) -> Vec<LockWaiter> {
        let mut ready = Vec::new();
        let mut still_blocked = Vec::new();

        for w in self.waiters.drain(..) {
            let blocked = self.segments.iter().any(|seg| {
                seg.owner != w.pid
                    && seg.overlaps(w.start, w.end)
                    && !matches!(
                        (w.lock_type, seg.lock_type),
                        (LockType::Shared, LockType::Shared)
                    )
            });
            if blocked {
                still_blocked.push(w);
            } else {
                ready.push(w);
            }
        }

        self.waiters = still_blocked;
        ready
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Public API
// ──────────────────────────────────────────────────────────────────────────────

/// Host-level table of POSIX advisory file locks.
///
/// One instance lives on each [`Host`] (as `file_lock_table: RefCell<FileLockTable>`),
/// mirroring the existing `futex_table` pattern.
#[derive(Default)]
pub struct FileLockTable {
    files: HashMap<FileKey, FileLockState>,
}

impl FileLockTable {
    pub fn new() -> Self {
        Self::default()
    }

    /// Attempt to acquire `[start, end)` with `lock_type` for `requester`.
    ///
    /// Returns `Ok(())` on success, or `Err(conflicting_owner)` if a
    /// conflicting lock is held by another process.
    pub fn try_lock(
        &mut self,
        key: FileKey,
        start: u64,
        end: u64,
        lock_type: LockType,
        requester: ProcessId,
    ) -> Result<(), ProcessId> {
        let state = self.files.entry(key).or_default();
        match state.find_conflict(start, end, lock_type, requester) {
            Some(conflict) => Err(conflict.owner),
            None => {
                state.acquire(start, end, lock_type, requester);
                Ok(())
            }
        }
    }

    /// Query whether a prospective lock would conflict (implements `F_GETLK`).
    ///
    /// Returns `None` if the lock *can* be acquired, or
    /// `Some((blocking_lock_type, blocking_pid))` if it cannot.
    pub fn query_lock(
        &self,
        key: FileKey,
        start: u64,
        end: u64,
        lock_type: LockType,
        requester: ProcessId,
    ) -> Option<(LockType, ProcessId)> {
        self.files
            .get(&key)?
            .find_conflict(start, end, lock_type, requester)
            .map(|seg| (seg.lock_type, seg.owner))
    }

    /// Release `[start, end)` held by `owner`.
    ///
    /// Returns any waiters that are now unblocked and must be rescheduled via
    /// `host.resume(waiter.pid, waiter.tid)`.
    pub fn unlock(
        &mut self,
        key: FileKey,
        start: u64,
        end: u64,
        owner: ProcessId,
    ) -> Vec<LockWaiter> {
        self.files
            .entry(key)
            .or_default()
            .release(start, end, owner)
    }

    /// Release **all** locks held by `owner` on `key`.
    ///
    /// Must be called on `close()` of the last fd referencing the file
    /// description, or on process exit.  Returns newly unblocked waiters.
    pub fn release_all(&mut self, key: FileKey, owner: ProcessId) -> Vec<LockWaiter> {
        self.unlock(key, 0, u64::MAX, owner)
    }

    /// Register `waiter` as blocked on `key`.
    ///
    /// Called immediately before returning `SyscallError::Blocked` from an
    /// `F_SETLKW` handler.  The waiter will be returned by the next
    /// [`unlock`](Self::unlock) / [`release_all`](Self::release_all) call when
    /// its conflict is resolved.
    pub fn add_waiter(&mut self, key: FileKey, waiter: LockWaiter) {
        self.files.entry(key).or_default().waiters.push(waiter);
    }
}
