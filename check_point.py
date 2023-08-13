class line:
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2


def onLine(l1, p):
    # Check whether p is on the line or not
    if (
            p[0] <= max(l1.p1[0], l1.p2[0])
            and p[0] >= min(l1.p1[0], l1.p2[0])
            and (p[1] <= max(l1.p1[1], l1.p2[1]) and p[1] >= min(l1.p1[1], l1.p2[1]))
    ):
        return True
    return False


def direction(a, b, c):
    val = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
    if val == 0:
        # Collinear
        return 0
    elif val < 0:
        # Anti-clockwise direction
        return 2
    # Clockwise direction
    return 1


def isIntersect(l1, l2):
    # Four direction for two lines and points of other line
    dir1 = direction(l1.p1, l1.p2, l2.p1)
    dir2 = direction(l1.p1, l1.p2, l2.p2)
    dir3 = direction(l2.p1, l2.p2, l1.p1)
    dir4 = direction(l2.p1, l2.p2, l1.p2)

    # When intersecting
    if dir1 != dir2 and dir3 != dir4:
        return True

    # When p2 of line2 are on the line1
    if dir1 == 0 and onLine(l1, l2.p1):
        return True

    # When p1 of line2 are on the line1
    if dir2 == 0 and onLine(l1, l2.p2):
        return True

    # When p2 of line1 are on the line2
    if dir3 == 0 and onLine(l2, l1.p1):
        return True

    # When p1 of line1 are on the line2
    if dir4 == 0 and onLine(l2, l1.p2):
        return True

    return False


def checkInside(poly, p):
    n = len(poly)
    # When polygon has less than 3 edge, it is not polygon
    if n < 3:
        return False

    # Create a point at infinity, y is same as point p
    exline = line(p, [9999, p[1]])
    count = 0
    i = 0
    while True:
        # Forming a line from two consecutive points of poly
        side = line(poly[i], poly[(i + 1) % n])
        if isIntersect(side, exline):
            # If side is intersects ex
            if (direction(side.p1, p, side.p2) == 0):
                return onLine(side, p)
            count += 1

        i = (i + 1) % n
        if i == 0:
            break

    # When count is odd
    return count & 1;