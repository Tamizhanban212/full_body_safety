import pygame as pg
import numpy as np
import math 

WIDTH, HEIGHT = 800, 600

class window():
    def __init__(self, width, height, title="Dangerfield Visualiser"):
        pg.init()
        self.width = width
        self.height = height
        self.screen = pg.display.set_mode((width, height))
        pg.display.set_caption(title)
        self.clock = pg.time.Clock()
        self.font = pg.font.SysFont("Arial", 18)

    def clear(self):
        self.screen.fill((0, 0, 0))

    def draw_circle(self, position, radius, color):
        pg.draw.circle(self.screen, color, position, radius)

    def draw_line(self, start_pos, end_pos, color, width=1):
        pg.draw.line(self.screen, color, start_pos, end_pos, width)

    def draw_text(self, text, position, color=(255, 255, 255)):
        text_surface = self.font.render(text, True, color)
        self.screen.blit(text_surface, position)

    def update(self):
        pg.display.flip()
        self.clock.tick(60)

    def quit(self):
        pg.quit()
    
class Dangerfields():
    def __init__(self):
        pass
    def calculate_cksdf_link(self,r, r_i, r_ip1, v_i, v_ip1, k1, k2, gamma):
        alpha1, beta1, gamma1, = r - r_i
        alpha2, beta2, gamma2 = -(r_ip1  - r_i)

        a = alpha2**2 + beta2**2 + gamma2**2
        b = 2*(alpha1*alpha2 + beta1*beta2 + gamma1*gamma2)
        c = alpha1**2 + beta1**2 + gamma1**2


        a1, b1, c1 = v_i
        a2, b2, c2 = v_ip1 - v_i 

        A = a2**2 + b2**2 + c2**2
        B = 2*(a1*a2 + b1*b2 + c1*c2)
        C = a1**2 + b1**2 + c1**2

        M = alpha2*a2 + beta2*b2 + gamma2*c2
        N = alpha1*a2 + alpha2*a1 + beta1*b2 + beta2*b1 + gamma1*c2 + gamma2*c1
        P = alpha1*a1 + beta1*b1 + gamma1*c1

        term1 = k1 * self.integrate_dist_term
        term2 = k2 * gamma * self.integrate_vel_term
        term3 = k2 * self.integrate_ang_term

        return term1 + term2 + term3

    def integrate_dist_term(self):
        a, b, c = self.a, self.b, self.c
        pass

    def integrate_vel_term(self):
        a, b, c = self.a, self.b, self.c
        A, B, C = self.A, self.B, self.C
        pass

    def integrate_ang_term(self):
        a, b, c = self.a , self.b, self.c
        M, N, P = self.M, self.N, self.P
        pass

    



        