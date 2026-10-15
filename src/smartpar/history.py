"""Method step 5 (recording): run history without locks.

Normal runs only read history.json and write a separate runs/run_<utc>_<pid>.json;
``smartpar merge`` folds run files into history.json with an exponential moving
average of durations. All files are written atomically (temp file + os.replace).

Not implemented yet (phase 2).
"""
