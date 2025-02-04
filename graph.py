import pygame
import numpy as np
import neural_network as nw


model = nw.Network()
model.load(r'neural_networks\digit_test')


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
            pixel_value /= 255
            s.append(pixel_value)
        array_pixel.append(s)
    return np.array(array_pixel)


def draw_prediction_result(probabilities):
    display.fill((200, 200, 200), (500, 50, 500, 500))  # Очистка области для вывода результата
    text_font = pygame.font.SysFont('arial', 20)

    # Отображение вероятностей для каждой цифры
    for i, prob in enumerate(probabilities):
        text_surface = text_font.render(f'{i}: {prob:.2%}', True, (0, 0, 0))
        display.blit(text_surface, (520, 60 + i * 40))

    pygame.display.update()


def draw_brush(mouse_position):
    # Основной квадрат
    brush_size = 20  # Размер квадрата
    pygame.draw.rect(display, (255, 255, 255), (mouse_position[0] - brush_size // 2, mouse_position[1] - brush_size // 2, brush_size, brush_size))

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

            elif 320 < mouse[0] < 320 + 150 and 500 < mouse[1] < 500 + 50:
                pygame.draw.rect(display, BUTTON_COLOR_ON, (320, 500, 150, 50))
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    ar = pygame.PixelArray(display)
                    ara = np.array(ar)
                    ar.close()
                    ans = one_pixel(ara[50:470, 50:470]).transpose()
                    ans = ans.flatten()
                    for i in ans:
                        print(i, end=', ')
                    print()
                    print()
                    prob = model.predict(ans)
                    draw_prediction_result(prob)

            else:
                pygame.draw.rect(display, BUTTON_COLOR_OFF, (50, 500, 150, 50))
                pygame.draw.rect(display, BUTTON_COLOR_OFF, (320, 500, 150, 50))

            text_button = pygame.font.SysFont('arial', 15)
            text_button_clear_surface = text_button.render('ОЧИСТИТЬ', True, (0, 0, 0))
            text_button_get_surface = text_button.render('ОПРЕДЕЛИТЬ ЦИФРУ', True, (0, 0, 0))

            display.blit(text_button_clear_surface, (85, 515))
            display.blit(text_button_get_surface, (330, 515))

            if event.type == pygame.MOUSEMOTION and event.buttons[0] == 1 and \
        70 <= mouse[0] <= 450 and 70 <= mouse[1] <= 450:
                draw_brush(mouse)

            pygame.display.update()
            clock.tick(FPS)

run_game()