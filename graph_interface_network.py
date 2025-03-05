import pygame
import numpy as np
import neural_network as nw
from PIL import Image


model = nw.Network()
model.load(r'MNIST_model')  # Путь до папки с весами нейросети


def crop_and_center(img, target_size=(28, 28)):
    img = img.astype(np.uint8)
    rows = np.any(img > 0, axis=1)
    cols = np.any(img > 0, axis=0)

    if not np.any(rows) or not np.any(cols):
        print("Ошибка: Пустое изображение после обрезки!")
        return np.zeros(target_size, dtype=np.uint8)

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    cropped = img[rmin:rmax+1, cmin:cmax+1]
    crop_height, crop_width = cropped.shape
    max_size = 20  
    scale = max_size / max(crop_height, crop_width)

    new_size = (int(crop_width * scale), int(crop_height * scale))
    resized_img = Image.fromarray(cropped).resize(new_size, Image.LANCZOS)

    final_img = np.zeros(target_size, dtype=np.uint8)
    top = (target_size[0] - new_size[1]) // 2
    left = (target_size[1] - new_size[0]) // 2
    final_img[top:top + new_size[1], left:left + new_size[0]] = np.array(resized_img)

    return final_img


BUTTON_COLOR_OFF = (100, 100, 150)
BUTTON_COLOR_ON = (150, 150, 100)
FPS = 180

pygame.init()
clock = pygame.time.Clock()

display = pygame.display.set_mode((1050, 650))
display.fill(color=(200, 200, 200))
pygame.display.set_caption('Drawing digits')

text_main = pygame.font.SysFont('arial', 20)
text_surface = text_main.render('Нарисуйте цифру', True, (0, 0, 0))

def new_surf():
    surf = pygame.Surface((28 * 15, 28 * 15))
    surf.fill((0, 0, 0))
    display.blit(surf, (50, 50))
    display.blit(text_surface, (185, 20))

    for i in range(0, 29):
        pygame.draw.aaline(display, (255, 255, 255), [50 + 15 * i, 50], [50 + 15 * i, 470])
        pygame.draw.aaline(display, (255, 255, 255), [50, 50 + 15 * i], [470, 50 + 15 * i])

def one_pixel(arr):
    array_pixel = []
    for i in range(28):
        row = arr[i*15:(i+1)*15, :]
        s = []
        for j in range(28):
            el = row[:, j*15:(j+1)*15]
            pixel_value = int((np.sum(el) / (15 * 15)) / 16777215 * 255)
            if pixel_value == 32:
                pixel_value = 0 
            s.append(pixel_value)
        array_pixel.append(s)
    return np.array(array_pixel)

def draw_prediction_result(probabilities):
    display.fill((200, 200, 200), (500, 50, 500, 500))
    text_font = pygame.font.SysFont('arial', 20)

    for i, prob in enumerate(probabilities):
        text_surface = text_font.render(f'{i}: {prob:.2%}', True, (0, 0, 0))
        display.blit(text_surface, (520, 60 + i * 40))

    pygame.display.update()

def predict_digit():
    ar = pygame.PixelArray(display)
    ara = np.array(ar)
    ar.close()
    ans = one_pixel(ara[50:470, 50:470]).transpose()
    ans = ans.flatten()
    prob = model.predict(crop_and_center(ans.reshape((28, 28))).reshape((784)).astype(np.float64) / 255)
    draw_prediction_result(prob)

def draw_brush(mouse_position):
    brush_size = 20  
    pygame.draw.rect(display, (255, 255, 255), (mouse_position[0] - brush_size // 2, mouse_position[1] - brush_size // 2, brush_size, brush_size))
    predict_digit()

new_surf()
pygame.display.update()

def run_game():
    flag = True

    while flag:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print('STOP')
                exit()

            mouse = pygame.mouse.get_pos()

            if 50 < mouse[0] < 50 + 150 and 500 < mouse[1] < 500 + 50:
                pygame.draw.rect(display, BUTTON_COLOR_ON, (50, 500, 150, 50))
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    new_surf()
            else:
                pygame.draw.rect(display, BUTTON_COLOR_OFF, (50, 500, 150, 50))

            text_button = pygame.font.SysFont('arial', 15)
            text_button_clear_surface = text_button.render('ОЧИСТИТЬ', True, (0, 0, 0))
            display.blit(text_button_clear_surface, (85, 515))

            if event.type == pygame.MOUSEMOTION and event.buttons[0] == 1 and \
        70 <= mouse[0] <= 450 and 70 <= mouse[1] <= 450:
                draw_brush(mouse)

            pygame.display.update()
            clock.tick(FPS)

run_game()
