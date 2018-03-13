from pywinauto.application import Application
from pywinauto.keyboard import SendKeys

import time

time.sleep(10)
i = 0
while True:
    i = i + 1
    SendKeys('{DOWN}')

    if i > 10:
        break
