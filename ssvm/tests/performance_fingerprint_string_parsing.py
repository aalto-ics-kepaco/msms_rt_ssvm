import re
import sqlite3
import time

import numpy as np


if __name__ == "__main__":
    t_str = 0.0
    t_re = 0.0
    t_str2 = 0.0

    for rep in range(1000):
        conn = sqlite3.connect("Massbank_test_db.sqlite")
        fps, = zip(
            *conn.execute("SELECT fingerprint FROM fingerprints_data "
                          "     WHERE name is 'estate_cnt' "
                          "     ORDER BY random() "
                          "     LIMIT 50").fetchall()
        )
        conn.close()

        X_str = np.zeros((len(fps), 79))
        s = time.time()
        for i, fp_str in enumerate(fps):
            for fp_d in fp_str.split(","):
                _d, _cnt = fp_d.split(":")
                X_str[i, int(_d)] = int(_cnt)
        e = time.time()
        t_str += (e - s)
        # print("String split: %.3f" % (t_str := (time.time() - s)))

        pat = re.compile(r"(\d+):(\d+),*")

        X_re = np.zeros((len(fps), 79))
        s = time.time()
        for i, fp_str in enumerate(fps):
            for _d, _cnt in pat.findall(fp_str):
                X_re[i, int(_d)] = int(_cnt)
        e = time.time()
        t_re += (e - s)
        # print("Reg-ex: %.3f" % (t_re := (time.time() - s)))

        assert np.array_equal(X_str, X_re)

        #### reformat the fps
        fps_ref = []
        for fp_str in fps:
            ds, cts = [], []
            for fp_d in fp_str.split(","):
                _d, _cnt = fp_d.split(":")
                ds.append(_d)
                cts.append(_cnt)
            fps_ref.append((",".join(ds), ",".join(cts)))

        X_str2 = np.zeros((len(fps), 79))
        s = time.time()
        for i, (ds, cts) in enumerate(fps_ref):
            for d, cnt in zip(map(int, ds.split(",")), map(int, cts.split(","))):
                X_str2[i, d] = cnt
        e = time.time()
        t_str2 += (e - s)
        # print("String split (2): %.3f" % (t_str2 := (time.time() - s)))

        assert np.array_equal(X_str, X_str2)

    print(t_str)
    print(t_re)
    print(t_str2)
