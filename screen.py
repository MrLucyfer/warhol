import sys
import pygame
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras

mnist = keras.models.load_model('saved_model/warhol_model')

mnist.summary()

pygame.init()

size = width, height = 280, 280

screen = pygame.display.set_mode(size)
black = 0, 0, 0

def draw_ball(pos):
    x, y = pos
    pygame.draw.circle(screen, (255, 255, 255), (x,y), 10)

def predict(surface):
    pixels = pygame.surfarray.array3d(surface)
    resized = np.mean(pixels, axis=2)
    resized = cv2.resize(resized, (28,28))
    resized = np.transpose(resized)
    
    resized = resized / 255
    resized = np.reshape(resized, (1,28,28,1))

    prediction = mnist.predict(resized)

    print('Your prediction is: {}'.format(np.argmax(prediction)))


is_pressed = False  

while True:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                predict(screen)
            elif event.key == pygame.K_r:
                #black the canvas
                screen.fill(black)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            is_pressed = True
        elif event.type == pygame.MOUSEBUTTONUP:
            is_pressed = False

    if is_pressed:
        draw_ball(pygame.mouse.get_pos())

    pygame.display.flip()

pygame.quit()