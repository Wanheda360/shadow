use linux_api::errno::Errno;
use linux_api::fcntl::{DescriptorFlags, FcntlCommand, OFlag};
use log::debug;
use shadow_shim_helper_rs::simulation_time::SimulationTime;
use shadow_shim_helper_rs::syscall_types::ForeignPtr;

use crate::core::work::task::TaskRef;
use crate::cshadow;
use crate::host::descriptor::{CompatFile, File, FileStatus};
use crate::host::file_lock_table::{FileKey, LockType, LockWaiter};
use crate::host::syscall::handler::{SyscallContext, SyscallHandler};
use crate::host::syscall::type_formatting::SyscallNonDeterministicArg;
use crate::host::syscall::types::SyscallError;

// ──────────────────────────────────────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Read a `libc::flock` struct from managed-process memory.
///
/// `arg` is the raw third argument to `fcntl(2)`, which for lock commands is a
/// user-space pointer to a `struct flock`.
fn read_flock(
    ctx: &SyscallContext,
    arg: std::ffi::c_ulong,
) -> Result<libc::flock, SyscallError> {
    let ptr = ForeignPtr::<libc::flock>::from(shadow_shim_helper_rs::syscall_types::SyscallReg::from(arg));
    // `memory_borrow().read()` copies one `T` out of the managed process's
    // address space.  The exact call signature depends on your Shadow version;
    // adjust if your MemoryManager uses a different read method.
    ctx.objs
        .process
        .memory_borrow()
        .read(ptr)
        .map_err(|_| Errno::EFAULT.into())
}

/// Write a `libc::flock` back into managed-process memory (used by `F_GETLK`
/// to report the conflicting lock).
fn write_flock(
    ctx: &mut SyscallContext,
    arg: std::ffi::c_ulong,
    fl: libc::flock,
) -> Result<(), SyscallError> {
    let ptr = ForeignPtr::<libc::flock>::from(shadow_shim_helper_rs::syscall_types::SyscallReg::from(arg));
    ctx.objs
        .process
        .memory_borrow_mut()
        .write(ptr, &fl)
        .map_err(|_| Errno::EFAULT.into())
}

/// Derive a stable [`FileKey`] for the file referred to by `desc`.
///
/// * **`CompatFile::New`** — returns the address of the inner `Arc`/`RootedRc`
///   allocation.  Duplicated / inherited descriptors share the same Arc, so
///   they share the same key.
///
/// * **`CompatFile::Legacy`** — ideally you would call into the C layer to
///   obtain the host fd, then call `fstat` to get `(st_dev, st_ino)` and
///   combine them into a `u64`.  Until that plumbing is in place, the raw
///   pointer to the `LegacyFile` object is used.  **This is correct for
///   `fork()`-inherited descriptors but NOT for two independent `open()` calls
///   on the same filesystem path.**
///
/// TODO: Replace the Legacy branch with inode-based identity once the C layer
/// exposes a `legacyfile_hostfd(LegacyFile*) -> int` (or equivalent) helper.
pub fn file_key_for_desc(desc: &crate::host::descriptor::Descriptor) -> FileKey {
    match desc.file() {
        CompatFile::New(d) => {
            // `d.inner_file()` returns a reference to the `Arc`/`RootedRc`
            // wrapping the `File`.  We use its allocation address as a stable
            // identity that is shared across `dup` / `fork`.
            std::ptr::from_ref(d.inner_file()) as u64
        }
        CompatFile::Legacy(lf) => {
            // TODO: get host fd from lf, fstat → combine (st_dev << 32 | st_ino)
            // For now use the LegacyFile pointer.
            std::ptr::from_ref(lf) as u64
        }
    }
}

/// Convert `flock` fields into an absolute `[start, end)` byte range.
///
/// `l_len == 0` means "to end of file", which we encode as `u64::MAX`.
/// A negative `l_len` shifts the start leftward and sets `end = original_start`.
///
/// Currently only `SEEK_SET` (`l_whence == 0`) is fully handled.  `SEEK_CUR`
/// and `SEEK_END` require knowing the current file offset / size, which is not
/// readily available here; they return `EINVAL` as a conservative placeholder.
/// BoltDB always uses `SEEK_SET`, so this is sufficient for the immediate goal.
fn flock_to_range(fl: &libc::flock) -> Result<(u64, u64), SyscallError> {
    const SEEK_SET: i16 = libc::SEEK_SET as i16;

    if fl.l_whence != SEEK_SET {
        // TODO: resolve SEEK_CUR / SEEK_END against the current file position.
        debug!(
            "fcntl lock with l_whence={} is not yet supported (only SEEK_SET=0)",
            fl.l_whence
        );
        return Err(Errno::EINVAL.into());
    }

    if fl.l_start < 0 {
        return Err(Errno::EINVAL.into());
    }

    let start = fl.l_start as u64;

    let end = if fl.l_len == 0 {
        u64::MAX // lock to end of file
    } else if fl.l_len < 0 {
        // Negative length: lock the range ending at `start`.
        let magnitude = (-fl.l_len) as u64;
        if magnitude > start {
            return Err(Errno::EINVAL.into());
        }
        let new_start = start - magnitude;
        // Return as (new_start, start) — caller receives (start, end) where
        // we repurpose start here.  We just return the adjusted pair directly.
        return Ok((new_start, start));
    } else {
        start.saturating_add(fl.l_len as u64)
    };

    Ok((start, end))
}

/// Wake every `waiter` in `waiters` by scheduling `host.resume` at simtime +0.
fn wake_waiters(ctx: &SyscallContext, waiters: Vec<LockWaiter>) {
    for w in waiters {
        let (pid, tid) = (w.pid, w.tid);
        let task = TaskRef::new(move |host| {
            host.resume(pid, tid);
        });
        ctx.objs
            .host
            .schedule_task_with_delay(task, SimulationTime::ZERO);
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Core lock-command handler
// ──────────────────────────────────────────────────────────────────────────────

/// Handle `F_SETLK`, `F_SETLKW`, `F_GETLK` (and their OFD equivalents) for a
/// single file descriptor.
///
/// Returns `Ok(0)` on success, or the appropriate `SyscallError`.
fn handle_lock_cmd(
    ctx: &mut SyscallContext,
    fd: std::ffi::c_uint,
    cmd: FcntlCommand,
    arg: std::ffi::c_ulong,
) -> Result<std::ffi::c_long, SyscallError> {
    let fl = read_flock(ctx, arg)?;

    // Decode lock type from flock.l_type.
    let l_type = fl.l_type;
    let lock_type = match l_type as i32 {
        libc::F_RDLCK => LockType::Shared,
        libc::F_WRLCK => LockType::Exclusive,
        libc::F_UNLCK => {
            // ── UNLOCK ────────────────────────────────────────────────────
            let (start, end) = flock_to_range(&fl)?;
            let key = {
                let desc_table =
                    ctx.objs.thread.descriptor_table_borrow(ctx.objs.host);
                let desc = SyscallHandler::get_descriptor(&desc_table, fd)?;
                file_key_for_desc(desc)
            };
            let pid = ctx.objs.process.id();
            let waiters = ctx
                .objs
                .host
                .file_lock_table_borrow_mut()
                .unlock(key, start, end, pid);
            wake_waiters(ctx, waiters);
            return Ok(0);
        }
        _ => return Err(Errno::EINVAL.into()),
    };

    let (start, end) = flock_to_range(&fl)?;

    let key = {
        let desc_table = ctx.objs.thread.descriptor_table_borrow(ctx.objs.host);
        let desc = SyscallHandler::get_descriptor(&desc_table, fd)?;
        file_key_for_desc(desc)
    };

    let pid = ctx.objs.process.id();
    let tid = ctx.objs.thread.id();

    match cmd {
        // ── F_GETLK / F_OFD_GETLK ─────────────────────────────────────────
        FcntlCommand::F_GETLK | FcntlCommand::F_OFD_GETLK => {
            match ctx
                .objs
                .host
                .file_lock_table_borrow()
                .query_lock(key, start, end, lock_type, pid)
            {
                None => {
                    // No conflict: set l_type = F_UNLCK and write back.
                    let mut out = fl;
                    out.l_type = libc::F_UNLCK as i16;
                    write_flock(ctx, arg, out)?;
                    Ok(0)
                }
                Some((blocking_type, blocking_pid)) => {
                    // Conflict: fill in the blocking lock's info and write back.
                    let mut out = fl;
                    out.l_type = match blocking_type {
                        LockType::Shared => libc::F_RDLCK as i16,
                        LockType::Exclusive => libc::F_WRLCK as i16,
                    };
                    // l_pid is the PID of the blocking process.
                    // ProcessId is i32-compatible; adjust if your type differs.
                    out.l_pid = i32::from(blocking_pid);
                    write_flock(ctx, arg, out)?;
                    Ok(0)
                }
            }
        }

        // ── F_SETLK / F_OFD_SETLK ────────────────────────────────────────
        FcntlCommand::F_SETLK | FcntlCommand::F_OFD_SETLK => {
            match ctx
                .objs
                .host
                .file_lock_table_borrow_mut()
                .try_lock(key, start, end, lock_type, pid)
            {
                Ok(()) => Ok(0),
                Err(_blocker) => {
                    // Non-blocking: return EACCES (POSIX) or EAGAIN (Linux alias).
                    Err(Errno::EACCES.into())
                }
            }
        }

        // ── F_SETLKW / F_OFD_SETLKW ──────────────────────────────────────
        FcntlCommand::F_SETLKW | FcntlCommand::F_OFD_SETLKW => {
            // Try to acquire.  If we were previously blocked and are being
            // re-run (Shadow re-executes the syscall on wakeup), this path
            // is taken again — if the lock is now free we proceed; if someone
            // else grabbed it first, we re-block.
            match ctx
                .objs
                .host
                .file_lock_table_borrow_mut()
                .try_lock(key, start, end, lock_type, pid)
            {
                Ok(()) => Ok(0),
                Err(_blocker) => {
                    // Register this thread as a waiter so that the unlock
                    // path can call `host.resume(pid, tid)` for us.
                    ctx.objs
                        .host
                        .file_lock_table_borrow_mut()
                        .add_waiter(key, LockWaiter { pid, tid, start, end, lock_type });

                    // Block until we are explicitly woken.  We use the
                    // simulation end time as the outer deadline so the thread
                    // is eventually freed even if the lock is never released
                    // (e.g. if the holder crashes without cleanup).
                    //
                    // `restartable = true` so that SA_RESTART causes the
                    // kernel to re-issue F_SETLKW after a signal.
                    Err(SyscallError::new_blocked_until(
                        ctx.objs.host.params.sim_end_time,
                        true,
                    ))
                }
            }
        }

        _ => unreachable!("handle_lock_cmd called with non-lock command"),
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// SyscallHandler impl
// ──────────────────────────────────────────────────────────────────────────────

impl SyscallHandler {
    log_syscall!(
        fcntl,
        /* rv  */ std::ffi::c_long,
        /* fd  */ std::ffi::c_uint,
        /* cmd */ FcntlCommand,
        /* arg */ SyscallNonDeterministicArg<std::ffi::c_ulong>,
    );

    pub fn fcntl(
        ctx: &mut SyscallContext,
        fd: std::ffi::c_uint,
        cmd: std::ffi::c_uint,
        arg: std::ffi::c_ulong,
    ) -> Result<std::ffi::c_long, SyscallError> {
        // NOTE: this function must NOT run the C syscall handler for any
        // command that modifies descriptor state.

        let legacy_syscall_fn =
            |ctx: &mut SyscallContext| Self::legacy_syscall(cshadow::syscallhandler_fcntl, ctx);

        let mut desc_table = ctx.objs.thread.descriptor_table_borrow_mut(ctx.objs.host);
        let desc = Self::get_descriptor_mut(&mut desc_table, fd)?;

        let Ok(cmd) = FcntlCommand::try_from(cmd) else {
            debug!("Bad fcntl command: {cmd}");
            return Err(Errno::EINVAL.into());
        };

        Ok(match cmd {
            // ── Lock commands ─────────────────────────────────────────────
            FcntlCommand::F_SETLK
            | FcntlCommand::F_SETLKW
            | FcntlCommand::F_OFD_SETLK
            | FcntlCommand::F_OFD_SETLKW
            | FcntlCommand::F_GETLK
            | FcntlCommand::F_OFD_GETLK => {
                // Both New and Legacy descriptors are handled by our Rust lock
                // table now.  Drop the descriptor-table borrow before entering
                // handle_lock_cmd (which needs to re-borrow it internally).
                drop(desc_table);
                return handle_lock_cmd(ctx, fd, cmd, arg);
            }

            // ── Everything below is unchanged from the original handler ───

            FcntlCommand::F_GETFL => {
                let file = match desc.file() {
                    CompatFile::New(d) => d,
                    CompatFile::Legacy(_) => {
                        drop(desc_table);
                        return legacy_syscall_fn(ctx);
                    }
                };
                let file = file.inner_file().borrow();
                let flags = file.status().as_o_flags() | file.mode().as_o_flags();
                flags.bits().into()
            }

            FcntlCommand::F_SETFL => {
                let file = match desc.file() {
                    CompatFile::New(d) => d,
                    CompatFile::Legacy(_) => {
                        drop(desc_table);
                        return legacy_syscall_fn(ctx);
                    }
                };

                let status = i32::try_from(arg).or(Err(Errno::EINVAL))?;
                let mut status = OFlag::from_bits(status).ok_or(Errno::EINVAL)?;
                status.remove(OFlag::O_RDONLY | OFlag::O_WRONLY | OFlag::O_RDWR | OFlag::O_PATH);
                status.remove(
                    OFlag::O_CLOEXEC
                        | OFlag::O_CREAT
                        | OFlag::O_DIRECTORY
                        | OFlag::O_EXCL
                        | OFlag::O_NOCTTY
                        | OFlag::O_NOFOLLOW
                        | OFlag::O_TMPFILE
                        | OFlag::O_TRUNC,
                );

                let mut file = file.inner_file().borrow_mut();
                let old_flags = file.status().as_o_flags();
                let update_mask = OFlag::O_APPEND
                    | OFlag::O_ASYNC
                    | OFlag::O_DIRECT
                    | OFlag::O_NOATIME
                    | OFlag::O_NONBLOCK;
                let status = (old_flags & !update_mask) | (status & update_mask);
                let (status, remaining) = FileStatus::from_o_flags(status);
                if !remaining.is_empty() {
                    return Err(Errno::EINVAL.into());
                }
                file.set_status(status);
                0
            }

            FcntlCommand::F_GETFD => desc.flags().bits().into(),

            FcntlCommand::F_SETFD => {
                let flags = i32::try_from(arg).or(Err(Errno::EINVAL))?;
                let flags = DescriptorFlags::from_bits(flags).ok_or(Errno::EINVAL)?;
                desc.set_flags(flags);
                0
            }

            FcntlCommand::F_DUPFD => {
                let min_fd = arg.try_into().or(Err(Errno::EINVAL))?;
                let new_desc = desc.dup(DescriptorFlags::empty());
                let new_fd = desc_table
                    .register_descriptor_with_min_fd(new_desc, min_fd)
                    .or(Err(Errno::EINVAL))?;
                new_fd.into()
            }

            FcntlCommand::F_DUPFD_CLOEXEC => {
                let min_fd = arg.try_into().or(Err(Errno::EINVAL))?;
                let new_desc = desc.dup(DescriptorFlags::FD_CLOEXEC);
                let new_fd = desc_table
                    .register_descriptor_with_min_fd(new_desc, min_fd)
                    .or(Err(Errno::EINVAL))?;
                new_fd.into()
            }

            FcntlCommand::F_GETPIPE_SZ => {
                let file = match desc.file() {
                    CompatFile::New(d) => d,
                    CompatFile::Legacy(_) => {
                        return legacy_syscall_fn(ctx);
                    }
                };
                if let File::Pipe(pipe) = file.inner_file() {
                    pipe.borrow().max_size().try_into().unwrap()
                } else {
                    return Err(Errno::EINVAL.into());
                }
            }

            cmd => {
                warn_once_then_debug!("Unhandled fcntl command: {cmd:?}");
                return Err(Errno::EINVAL.into());
            }
        })
    }
}