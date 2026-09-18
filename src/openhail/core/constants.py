# dictionary keys
LOC = "loc"
Q = "Q"
TIME = "time"

ORIG = "orig"
DEST = "dest"
Q_DEST = "q_dest"
T_DEST = "t_dest"

NEXT_EPOCH = "next_epoch"
CHARGER = "used_charger"
TARGET_CHARGER = "target_charger"
WAYPOINT = "waypoint"
TYPE = "type"

PROC_TIME = "process_time"

# charger observation keys
CHARGERS = "chargers"
CAPACITY = "capacity"
OCCUPANCY = "occupancy"
QUEUE = "queue"

# decision metadata keys
EPOCH_TYPE = "epoch_type"

# job values
JOB_SERVE = 10
JOB_REPO = 4
JOB_GO_CHARGE = 3
JOB_QUEUE = 2
JOB_CHARGE = 1
JOB_IDLE = 0
JOB_NULL = -1

# non instance specific parameters
EPS = 1e-3

NULL_EPOCH = -1
END_OF_SERVE = 0
END_OF_CHARGE = 1
END_OF_REPO = 2
NEW_REQUEST = 3
CLOCK_EPOCH = 4
REQUESTED_EPOCH = 5
END_OF_HORIZON = 6

EV_SUMMARY_FULL = 0
EV_SUMMARY_CLOCK = 1
EV_SUMMARY_NONE = 2
