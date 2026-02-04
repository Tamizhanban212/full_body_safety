import pygame as pg
import numpy as np
import math 
from scipy.integrate import quad

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

    def compute_cdf_integral(a, b, c, A, B, C, M, N, P, k1=1.0, k2=1.0, gamma=1.0):
        """
        Computes:
        CDF = k1 ∫₀¹ dt / √(a t² + b t + c)
            + k2 γ ∫₀¹ √(A t² + B t + C) dt ⋅ ∫₀¹ (M t² + N t + P) dt
            + k2 ∫₀¹ dt / (a t² + b t + c)
            + k2 ∫₀¹ dt / (a t² + b t + c)^{3/2}
        """
        # First integral: 1 / sqrt(quadratic)
        def integrand1(t):
            return 1.0 / np.sqrt(a*t**2 + b*t + c)
        I1, _ = quad(integrand1, 0, 1)

        # Second part - product of two integrals
        def integrand_sqrt(t):
            return np.sqrt(A*t**2 + B*t + C)
        I_sqrt, _ = quad(integrand_sqrt, 0, 1)

        def integrand_linear(t):
            return M*t**2 + N*t + P
        I_lin, _ = quad(integrand_linear, 0, 1)

        cross_term = I_sqrt * I_lin

        # Third: 1 / quadratic
        def integrand3(t):
            return 1.0 / (a*t**2 + b*t + c)
        I3, _ = quad(integrand3, 0, 1)

        # Fourth: 1 / (quadratic)^{3/2}
        def integrand4(t):
            q = a*t**2 + b*t + c
            return 1.0 / (q * np.sqrt(q))
        I4, _ = quad(integrand4, 0, 1)

        # Combine
        result = (
            k1 * I1 +
            k2 * gamma * cross_term +
            k2 * I3 +
            k2 * I4
        )

        return result

    



        