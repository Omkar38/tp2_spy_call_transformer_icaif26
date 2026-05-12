# Configuration

Runtime defaults live in [src/tp2rr2/config.py](../src/tp2rr2/config.py).

The important knobs to tune on validation only are:

- prediction threshold
- minimum normalized severity
- max leg and average spread percentage
- minimum volume/open interest
- max trades per day
- max contract overlap per day
- detection strike/expiration caps

Do not tune these on the 2024 test set.
