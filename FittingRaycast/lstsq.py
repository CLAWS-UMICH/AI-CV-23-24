import numpy as np

with open('points-to-fit.txt', 'r') as f:
    source_matrix_pre = []
    target_matrix_pre = []

    for line in f.readlines():
        source, target = line.strip().split("->")
        source, target = source.strip(), target.strip()
        sx,sy,tx,ty = source.split(',') + target.split(',')
        sx,sy,tx,ty = int(sx),int(sy),int(tx),int(ty)
        source_matrix_pre.append([sx, sy, 1])
        target_matrix_pre.append([tx, ty, 1])
    
    source_m = np.array(source_matrix_pre)
    target_m = np.array(target_matrix_pre)

    # Compute the transformation matrix using least squares
    M = np.linalg.lstsq(source_m, target_m, rcond=None)[0]

    print("Transformation matrix:")
    print(M)
    np.save("fit_raycast_matrix", M)

    print("Calculated Targets")
    for x,y,_ in source_matrix_pre:
        print(np.dot(np.array([x,y,1]),M))