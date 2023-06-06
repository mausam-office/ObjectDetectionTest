# clustering algorithm


import numpy as np


def split_points_old(points, num_row, h):
    rows = [{} for _ in range(num_row)]
    h_mid = int(h/num_row)
    h2_mid = h_mid*2
    for point in points:
        value, key = list(point.keys())[0], list(point.values())[0]

        if value[1] < h_mid:
            rows[0][value] = key

        elif value[1] < h2_mid:
            rows[1][value] = key

        else:
            rows[2][value] = key
    return rows

def split_points(points, num_rows, peaks_minima, h):
    num_split_pnt = len(peaks_minima)
    rows = [{} for _ in range(4)]
    peaks_minima = [val for val in peaks_minima]
    # print('peaks_minima: ', peaks_minima, type(peaks_minima))
    # peaks_minima = [46]


    if num_rows <= 1:
        # print("NUmber of rows found is 0 or 1")
        for point in points:
            value, key = list(point.keys())[0], list(point.values())[0]
            rows[0][value] = key

    elif num_rows>1 and num_split_pnt>=1:
        # print("in else if")
        h1 = peaks_minima[0] if len(peaks_minima) > 0 else 0
        h2 = peaks_minima[1] if len(peaks_minima) > 1 else 0
        if not h1 and not h1:
            num_split_pnt = 1
            h1 = int(h/2)
        # print('h1, h2', h1, h2)
        for point in points:
            value, key = list(point.keys())[0], list(point.values())[0]
            # print(value, key)

            if value[1] <= h1 and num_split_pnt >= 1:
                rows[0][value] = key
            elif value[1] <= h2 and num_split_pnt >= 2:
                rows[1][value] = key
            else:
                rows[2][value] = key

    else:
        # print("No split points found ")
        for point in points:
            value, key = list(point.keys())[0], list(point.values())[0]
            rows[0][value] = key
    return rows


def sort_points(rows):
    plate_val = []
    for row in rows:
        # print('row', row)
        plate_val.extend([val.capitalize() for key, val in sorted(row.items(), key=lambda ele: ele[0])])
        # ordered_vals = [val for key, val in sorted(row.items(), key=lambda ele: ele[0])]
        # ordered_vals = ''.join(ordered_vals)
        # print('plate_val', plate_val)

    return ''.join(plate_val)


if __name__=="__main__":
    h, w = 375, 500 
    num_row = 1
    
    points = [{(513, 237): '5'}, {(197, 286): 'ba'}, {(321, 277): '1'}, {(624, 214): '8'}, {(747, 195): '5'}, {(394, 256): 'pa'}]
    
    # rows = split_points(points, num_row, h)
    rows = split_points(points, np.array([150]))
    print(rows)
    plate_val = sort_points(rows)
    print('row points', plate_val)
    
    
