import numpy as np
from scipy.integrate import quad


class Dangerfields():
    def __init__(self):
        self.M  = 0
        self.N = 0
        self.P = 0
        self.a = 0
        self.b = 0
        self.c = 0
        self.A = 0
        self.B = 0
        self.C = 0

    def calculate_cksdf_link(self,r, r_i, r_ip1, v_i, v_ip1, k1, k2, gamma):
        alpha1, beta1, gamma1, = r - r_i
        alpha2, beta2, gamma2 = -(r_ip1  - r_i)

        self.a = alpha2**2 + beta2**2 + gamma2**2
        self.b = 2*(alpha1*alpha2 + beta1*beta2 + gamma1*gamma2)
        self.c = alpha1**2 + beta1**2 + gamma1**2


        a1, b1, c1 = v_i
        a2, b2, c2 = v_ip1 - v_i 

        self.A = a2**2 + b2**2 + c2**2
        self.B = 2*(a1*a2 + b1*b2 + c1*c2)
        self.C = a1**2 + b1**2 + c1**2

        self.M = alpha2*a2 + beta2*b2 + gamma2*c2
        self.N = alpha1*a2 + alpha2*a1 + beta1*b2 + beta2*b1 + gamma1*c2 + gamma2*c1
        self.P = alpha1*a1 + beta1*b1 + gamma1*c1

        # term1 = k1 * self.integrate_dist_term
        # term2 = k2 * gamma * self.integrate_vel_term
        # term3 = k2 * self.integrate_ang_term

        # return term1 + term2 + term3
        output = self.compute_cdf_integral(
            self.a, self.b, self.c,
            self.A, self.B, self.C,
            self.M, self.N, self.P,
            k1=k1, k2=k2, gamma=gamma
        )
        return output
    def compute_cdf_integral(self, a, b, c, A, B, C, M, N, P, k1=1.0, k2=1.0, gamma=1.0):
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

    



        