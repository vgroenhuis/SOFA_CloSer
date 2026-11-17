"""Product catalogue of the snack shelf (photos 1-5, 1126 x 2000): photo crop (quad in the photo's pixels, top-left,
top-right, bottom-right, bottom-left) + size in cm (w across, h up, d deep). Sizes are estimates scaled from the
photos (a Pringles can: 7.5 cm across, 23.5 cm tall).
kind: can (a round tube) | box (a paper tin, square) | tub (a round plastic tub) | bag (deformable in the simulation)
"""
P = {}
def p(id, name, img, quad, w, h, d, kind='box', src='photo'):
    P[id] = dict(id=id, name=name, img=img, quad=quad, w=w, h=h, d=d, kind=kind, src=src)

# ---- RIGHT BAY (chips) ---------------------------------------------------
# R1: Pringles, a pyramid
p('prg_orig', 'Pringles Original', '5', [(383, 540), (446, 540), (447, 732), (384, 732)], 7.5, 23.5, 7.5, 'can')
p('prg_papr', 'Pringles Paprika', '5', [(628, 540), (685, 540), (686, 730), (629, 730)], 7.5, 23.5, 7.5, 'can')
p('prg_sco', 'Pringles Sour Cream & Onion', '5', [(688, 540), (746, 540), (746, 728), (689, 728)], 7.5, 23.5, 7.5, 'can')
p('prg_hot', 'Pringles Hot & Spicy', '5', [(876, 537), (919, 537), (920, 718), (877, 718)], 7.5, 23.5, 7.5, 'can')
p('prg_s_orig', 'Pringles Original 70 g', '5', [(982, 628), (1038, 628), (1038, 712), (982, 712)], 7.5, 11.5, 7.5, 'can')
p('prg_s_papr', 'Pringles Paprika 70 g', '5', [(1028, 535), (1080, 535), (1080, 620), (1028, 620)], 7.5, 11.5, 7.5, 'can')
# R2: paper tins
p('lays_max', "Lay's Max paprika", '5', [(452, 806), (519, 806), (520, 958), (454, 960)], 9, 19.5, 9)
p('bugles', "Lay's Bugles nacho cheese", '5', [(586, 802), (651, 802), (652, 952), (587, 954)], 9, 19.5, 9)
p('doritos_tin', 'Doritos nacho cheese (tin)', '5', [(730, 800), (804, 800), (805, 946), (731, 948)], 9.5, 19.5, 9.5)
p('wavy_ched', "Lay's Wavy mature cheddar & onion", '5', [(928, 782), (996, 782), (996, 922), (929, 922)], 9.5, 19.5, 9.5)
p('wavy_ham', "Lay's Wavy roasted ham", '5', [(1018, 784), (1092, 784), (1092, 925), (1019, 925)], 9.5, 19.5, 9.5)
# R3: tubes and Doritos bags
p('tube_red', 'Potato crisps original (tube)', '2', [(410, 1180), (473, 1180), (474, 1322), (411, 1322)], 7, 23, 7, 'can')
p('tube_green', 'Potato crisps sour cream (tube)', '2', [(650, 1200), (718, 1200), (719, 1378), (651, 1378)], 7, 23, 7, 'can')
p('tube_orange', 'Spar Potato Crisps paprika', '2', [(880, 1236), (966, 1236), (967, 1440), (881, 1442)], 7.5, 23, 7.5, 'can')
p('doritos_bag', "Doritos Flamin' Hot", '4', [(928, 1108), (1052, 1108), (1052, 1280), (928, 1280)], 19, 21, 6, 'bag')
# R4: tortilla chips
p('gwoon_nat', 'Gwoon tortilla chips naturel', '2', [(655, 1495), (775, 1492), (777, 1670), (657, 1673)], 21, 19, 7, 'bag')
p('gwoon_nacho', 'Gwoon tortilla chips nacho cheese', '4', [(890, 1392), (1030, 1392), (1032, 1525), (890, 1525)], 21, 19, 7, 'bag')
# R5: Chio
p('chio', 'Chio chips', '2', [(800, 1660), (1060, 1665), (1060, 1760), (800, 1755)], 22, 17, 6, 'bag')
# R6 (the deck): Takis, Breakers
p('takis_nacho', 'Takis Intense Nacho', '2', [(331, 1680), (420, 1678), (423, 1822), (334, 1824)], 15, 21, 5, 'bag')
p('takis_blue', 'Takis Blue Heat', '2', [(508, 1702), (598, 1700), (600, 1865), (510, 1868)], 15, 21, 5, 'bag')
p('takis_fuego', 'Takis Fuego', '2', [(615, 1733), (707, 1731), (708, 1868), (616, 1869)], 15, 21, 5, 'bag')
p('breakers', 'Breakers paprika', '2', [(800, 1780), (935, 1780), (938, 1935), (802, 1938)], 20, 21, 7, 'bag')

# ---- LEFT BAY (nuts) -----------------------------------------------------
p('t_chocopinda', "Gwoon chocolade pinda's", '5', [(70, 648), (180, 648), (180, 688), (70, 688)], 11, 6.5, 11, 'tub')
p('t_choco', 'Gwoon choco melange', '5', [(200, 690), (306, 690), (306, 733), (200, 733)], 11, 6.5, 11, 'tub')
p('t_pinda', "Gwoon pinda's", '5', [(3, 870), (102, 870), (102, 925), (3, 925)], 11, 6.5, 11, 'tub')
p('t_cashew', 'Gwoon cashewnoten', '5', [(108, 874), (218, 874), (218, 925), (108, 925)], 11, 6.5, 11, 'tub')
p('t_pistache', 'Gwoon pistachenoten', '5', [(228, 870), (325, 870), (325, 920), (228, 920)], 11, 6.5, 11, 'tub')
p('t_kalpang', 'Gwoon katjang pedis', '5', [(6, 1085), (103, 1085), (103, 1133), (6, 1133)], 11, 6.5, 11, 'tub')
p('t_macadamia', 'Gwoon macadamia', '5', [(225, 1082), (325, 1082), (325, 1135), (225, 1135)], 11, 6.5, 11, 'tub')
p('b_crackers', 'Zoute crackers', '5', [(95, 1208), (188, 1205), (188, 1322), (95, 1325)], 15, 19, 6, 'bag')
p('b_borrelmix', 'Borrelmix', '5', [(188, 1224), (269, 1224), (269, 1311), (188, 1311)], 14, 17, 6, 'bag')
p('b_bbq', 'BBQ paprika borrelnoten', '5', [(230, 1362), (340, 1362), (341, 1472), (231, 1474)], 16, 18, 6, 'bag')
p('b_peanuts', "Gwoon pinda's in de dop", '5', [(65, 1535), (285, 1535), (288, 1698), (66, 1698)], 25, 19, 9, 'bag')

# ---- LAYOUT --------------------------------------------------------------
# Two bays: the left one 100 cm wide (nuts), the right one 125 cm (chips). Shelf surface heights (cm), top to bottom.
BAYS = [('L', 0.0, 100.0), ('R', 100.0, 125.0)]
LEFT_Y = [126, 102, 78, 57, 36, 12]
RIGHT_Y = [126, 102, 77, 56, 36, 12]

def row(*g): return list(g)
def g(pid, n=1, rows=None, gap=0, back=0, stack=1, price=None, yellow=False, tag=True):
    return dict(pid=pid, n=n, rows=rows, gap=gap, back=back, stack=stack, price=price, yellow=yellow, tag=tag)

LEFT = [
 row(g('t_chocopinda', 3, stack=3, gap=1, price='4.49'), g('t_choco', 3, stack=3, gap=1, price='3.89')),
 row(g('t_pinda', 3, stack=2, gap=1, price='4.49'), g('t_cashew', 2, stack=2, gap=.5, price='3.99'), g('t_pistache', 3, stack=2, gap=.5, price='3.99')),
 row(g('t_kalpang', 3, stack=3, gap=1, price='2.89'), g('t_macadamia', 3, stack=3, gap=.5, price='5.49'), g('t_pistache', 2, stack=3, gap=.5, price='3.99')),
 row(g('b_crackers', 3, gap=1, price='4.49'), g('b_borrelmix', 3, gap=1, price='2.29')),
 row(g('b_bbq', 5, gap=1, price='1.75')),
 row(g('b_peanuts', 3, gap=2, price='3.49')),
]
RIGHT = [
 row(g('prg_orig', 3, stack=2, gap=1, price='3.09'), g('prg_papr', 2, stack=3, price='3.09'), g('prg_papr', 1, stack=2, tag=False),
     g('prg_sco', 2, stack=2, price='3.09'), g('prg_hot', 2, stack=2, price='3.09'),
     g('prg_s_orig', 2, stack=2, gap=1, price='2.15'), g('prg_s_papr', 2, stack=2, price='2.15')),
 row(g('lays_max', 2, gap=1, price='2.75'), g('bugles', 2, gap=.5, price='3.29'), g('doritos_tin', 3, gap=.5, price='2.99'),
     g('wavy_ched', 3, gap=.5, price='2.69'), g('wavy_ham', 2, gap=.5, price='2.69')),
 row(g('tube_red', 2, gap=1, price='2.29'), g('tube_green', 2, gap=1, price='2.15'), g('tube_orange', 4, gap=1, price='2.15'),
     g('doritos_bag', 3, gap=2, price='2.79')),
 row(g('gwoon_nat', 3, gap=2, price='1.25'), g('gwoon_nacho', 2, gap=3, price='1.25')),
 row(g('chio', 5, gap=1.5, price='0.95')),
 row(g('takis_nacho', 2, gap=1.5, price='3.69'), g('takis_blue', 2, gap=1, price='3.69'), g('takis_fuego', 2, gap=1, price='3.69'),
     g('breakers', 1, gap=3, price='2.29')),
]
