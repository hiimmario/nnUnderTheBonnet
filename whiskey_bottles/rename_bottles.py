import os
from datetime import datetime
import shutil

startTime = datetime.now()

# pictures filepath
base = r'C:\Users\Mario\PycharmProjects\nn_bottles\bottles'

for filename in os.listdir(base):
    path = os.path.join(base, filename)

    '''

    0==Ardbeg
    1==Laphroaig
    2==Bowmore
    3==Bruichladdich
    4==Bunnahabhain
    5==Caol Ila
    6==Kilchoman
    7==Lagavulin
    8==Port Ellen
    9==Port Askaig

    '''

    if filename.startswith("Ardbeg_"):
        os.rename(path, path.replace("Ardbeg_", "0_"))

    elif filename.startswith("Laphroaig_"):
        os.rename(path, path.replace("Laphroaig_", "1_"))

    elif filename.startswith("Bowmore_"):
        os.rename(path, path.replace("Bowmore_", "2_"))

    elif filename.startswith("Bruichladdich_"):
        os.rename(path, path.replace("Bruichladdich_", "3_"))

    elif filename.startswith("Bunnahabhain_"):
        os.rename(path, path.replace("Bunnahabhain_", "4_"))

    elif filename.startswith("Caol Ila_"):
        os.rename(path, path.replace("Caol Ila_", "5_"))

    elif filename.startswith("Kilchoman_"):
        os.rename(path, path.replace("Kilchoman_", "6_"))

    elif filename.startswith("Lagavulin_"):
        os.rename(path, path.replace("Lagavulin_", "7_"))

    elif filename.startswith("Port Ellen_"):
        os.rename(path, path.replace("Port Ellen_", "8_"))

    elif filename.startswith("Port Askaig_"):
        os.rename(path, path.replace("Port Askaig_", "9_"))

print(datetime.now() - startTime)