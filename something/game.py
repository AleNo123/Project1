import pygame, sys, random
pygame.init()
from Proba import *
import threading

def user():
    stream_info = pylsl.resolve_stream()
    inlet = StreamInlet(stream_info[0], max_chunklen=750)
    model = neural_network_training()
    exist = True
    while exist:
        chunk, t_ = inlet.pull_chunk(timeout=3)
        chunk = chunk[:650]
        x = np.expand_dims(chunk, axis=0)
        res = model.predict(x)
        if np.where(res[0] == max(res[0]))[0][0] == 0:
            if r[1] > 0:
                pygame.draw.rect(s, (0, 0, 0), r)
                r[1] -= xod
                pygame.draw.rect(s, (200, 100, 55), r)
        if np.where(res[0] == max(res[0]))[0][0] == 1:
            if r[1] < 600:
                pygame.draw.rect(s, (0, 0, 0), r)
                r[1] += xod
                pygame.draw.rect(s, (200, 100, 55), r)
        print("user "+ str(np.where(res[0] == max(res[0]))[0][0]))

def user1():
    stream_info = pylsl.resolve_stream()
    inlet = StreamInlet(stream_info[1], max_chunklen=750)
    model = neural_network_training()
    exist = True
    while exist:
        chunk, t_ = inlet.pull_chunk(timeout=3)
        chunk = chunk[:650]
        x = np.expand_dims(chunk, axis=0)
        res = model.predict(x)
        if np.where(res[0] == max(res[0]))[0][0] == 0:
            if r[1] > 0:
                pygame.draw.rect(s, (0, 0, 0), r1)
                r1[1] -= xod
                pygame.draw.rect(s, (200, 100, 55), r1)
        if np.where(res[0] == max(res[0]))[0][0] == 1:
            if r[1] < 600:
                pygame.draw.rect(s, (0, 0, 0), r1)
                r1[1] += xod
                pygame.draw.rect(s, (200, 100, 55), r1)
        print("user1 " + str(np.where(res[0] == max(res[0]))[0][0]))

stream_info = pylsl.resolve_stream()
print(stream_info)
inlet1 = StreamInlet(stream_info[1], max_chunklen=750)
# thr = threading.Thread(target=user, args=())
# thr1 = threading.Thread(target=user1, args=())
# thr.start()
# thr1.start()

# ---------------------------------
font = pygame.font.SysFont('franklingothicmediumcond', 58)


a0 = 1
a1 = 1
w = 1500
h = 800
wr = 5
hr = 200
rad = 10
count0 = random.choice((-a0, a0))
count1 = random.choice((-a0, a0))
xod = 60
score1 = 0
score0 = 0


s = pygame.display.set_mode((w, h))
clock = pygame.time.Clock()
r = [0, h / 2 - hr / 2, wr, hr]
r1 = [w - wr, h / 2 - hr / 2, wr, hr]
r2 = [748, 20, 4, 40]
c = [w / 2, h / 2]
pygame.draw.rect(s, (200, 100, 55), r)
pygame.draw.rect(s, (200, 100, 55), r1)
pygame.draw.circle(s, (200, 100, 55), c, rad)
for i in range(10):
    pygame.draw.rect(s, (200, 100, 55), r2)
    r2[1] += 80
r2 = [748, 20, 4, 40]
f = font.render(str(score0), 1, (255, 255, 255), (0, 0, 0))
s.blit(f, (700, 5))
f = font.render(str(score1), 1, (255, 255, 255), (0, 0, 0))
s.blit(f, (770, 5))

thr = threading.Thread(target=user, args=())
thr1 = threading.Thread(target=user1, args=())
thr.start()
thr1.start()

while 1:
    clock.tick(250)
    global exist
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exist = False
            sys.exit()
        # if event.type == pygame.KEYDOWN and event.key == pygame.K_w:
        #     if r[1] > 0:
        #         pygame.draw.rect(s, (0, 0, 0), r)
        #         r[1] -= xod
        #         pygame.draw.rect(s, (200, 100, 55), r)
        # if event.type == pygame.KEYDOWN and event.key == pygame.K_s:
        #     if r[1] < 600:
        #         pygame.draw.rect(s, (0, 0, 0), r)
        #         r[1] += xod
        #         pygame.draw.rect(s, (200, 100, 55), r)
        # if event.type == pygame.KEYDOWN and event.key == pygame.K_o:
        #     if r[1] > 0:
        #         pygame.draw.rect(s, (0, 0, 0), r1)
        #         r1[1] -= xod
        #         pygame.draw.rect(s, (200, 100, 55), r1)
        # if event.type == pygame.KEYDOWN and event.key == pygame.K_l:
        #     if r[1] < 600:
        #         pygame.draw.rect(s, (0, 0, 0), r1)
        #         r1[1] += xod
        #         pygame.draw.rect(s, (200, 100, 55), r1)
        if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            count0 = random.choice((-a0, a0))
            count1 = random.choice((-a1, a1))
            score1 = 0
            pygame.draw.rect(s, (0, 0, 0), [800, 0, 900, 60])
            f = font.render(str(score1), 1, (255, 255, 255), (0, 0, 0))
            s.blit(f, (770, 5))
            score0 = 0
            pygame.draw.rect(s, (0, 0, 0), [100, 0, 600, 60])
            f = font.render(str(score0), 1, (255, 255, 255), (0, 0, 0))
            s.blit(f, (700, 5))
            f = font.render('Winner', 1, (0, 0, 0), (0, 0, 0))
            s.blit(f, (900, 5))
            f = font.render('Winner', 1, (0, 0, 0), (0, 0, 0))
            s.blit(f, (300, 5))
            pygame.draw.rect(s, (0, 0, 0), r1)
            r1[1] = h / 2 - hr / 2
            pygame.draw.rect(s, (200, 100, 55), r1)
            pygame.draw.rect(s, (0, 0, 0), r)
            r[1] = h / 2 - hr / 2
            pygame.draw.rect(s, (200, 100, 55), r)
            pygame.draw.circle(s, (0, 0, 0), c, rad)
            c = [w / 2, h / 2]
    if c[1] == h - rad:
        count1 = -a1
    elif c[1] == rad:
        count1 = a1
    elif c[0] == w - rad - wr:
        if c[1] > r1[1] and (c[1] < r1[1] + hr):
            count0 = -a0
        else:
            pygame.draw.circle(s, (0, 0, 0), c, rad)
            c = [w / 2, h / 2]
            count0 = random.choice((-a0, a0))
            count1 = random.choice((-a1, a1))
            score0 += 1
            if score0 != 10:
                f = font.render(str(score0), 1, (255, 255, 255), (0, 0, 0))
                s.blit(f, (700, 5))
            else:
                f = font.render(str(score0), 1, (255, 255, 255), (0, 0, 0))
                s.blit(f, (670, 5))
                f = font.render('Winner', 1, (255, 255, 255), (0, 0, 0))
                s.blit(f, (300, 5))
                pygame.draw.circle(s, (0, 0, 0), c, rad)
                c = [w / 2, h / 2]
                count0 = 0
                count1 = 0
    elif c[0] == rad + wr:
        if c[1] > r[1] and (c[1] < r[1] + hr):
            count0 = a0
        else:
            pygame.draw.circle(s, (0, 0, 0), c, rad)
            c = [w / 2, h / 2]
            count0 = random.choice((-a0, a0))
            count1 = random.choice((-a1, a1))
            score1 += 1
            if score1 != 10:
                f = font.render(str(score1), 1, (255, 255, 255), (0, 0, 0))
                s.blit(f, (770, 5))
            else:
                f = font.render(str(score1), 1, (255, 255, 255), (0, 0, 0))
                s.blit(f, (770, 5))
                f = font.render('Winner', 1, (255, 255, 255), (0, 0, 0))
                s.blit(f, (900, 5))
                pygame.draw.circle(s, (0, 0, 0), c, rad)
                c = [w / 2, h / 2]
                count0 = 0
                count1 = 0
    pygame.draw.circle(s, (0, 0, 0), c, rad)
    c[0] += count0
    c[1] += count1
    pygame.draw.circle(s, (200, 100, 55), c, rad)
    for i in range(10):
        pygame.draw.rect(s, (200, 100, 55), r2)
        r2[1] += 80
    r2 = [748, 20, 4, 40]
    pygame.display.update()