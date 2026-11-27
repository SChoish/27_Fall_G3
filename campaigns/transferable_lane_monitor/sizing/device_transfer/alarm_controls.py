"""Run additional comparisons when each primary circuit finishes."""
import fcntl

import alarm_pilot as pilot

if __name__ == "__main__":
    args=pilot.exp.arguments()
    with (args.out/"alarm_controls.lock").open("a") as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(0)
        pilot.METHODS=("role40","keep_head","reset_head","prefix_attention")
        pilot.main(args,wait_for_primary=True)
