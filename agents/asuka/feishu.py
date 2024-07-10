import pygame

# 初始化 pygame
pygame.init()

# 设置窗口大小
screen_width = 640
screen_height = 480
screen = pygame.display.set_mode((screen_width, screen_height))

# 设置标题
pygame.display.set_caption("贪吃蛇游戏")

# 定义蛇的属性
snake = Snake(screen)

# 定义食物的属性
food = Food()

# 主循环
while True:
    # 处理事件
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # 更新蛇的位置
    snake.update()

    # 检查蛇是否碰到食物
    if snake.collide(food):
        # 生成新的食物
        food = Food()
        # 增加蛇的长度
        snake.grow()

    # 更新画面
    screen.fill((255, 255, 255))
    snake.draw(screen)
    food.draw(screen)
    pygame.display.flip()