"""Run additional comparisons after the main ledger has completed."""
import alarm_pilot as pilot

if __name__ == "__main__":
    pilot.METHODS=("role40","keep_head","reset_head","prefix_attention")
    pilot.main()
