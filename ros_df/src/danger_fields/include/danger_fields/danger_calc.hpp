#pragma once
#include <Eigen/Eigen>
#include <cmath>
#include <vector>
#include <algorithm>

class DangerField {
public:

    struct LinkState {
        Eigen::Vector3d r_i;
        Eigen::Vector3d r_ip1;
        Eigen::Vector3d v_i;
        Eigen::Vector3d v_ip1;
    };

    DangerField(double k1, double k2, double gamma, int n_pts)
        : k1_(k1), k2_(k2), gamma_(gamma), n_pts_(n_pts) {}

    // ── Single point danger ───────────────────────────────────────────────────
    // Evaluates the danger integrand at a single point r_s with velocity v_s
    double pointDanger(const Eigen::Vector3d& r_obs,
                       const Eigen::Vector3d& r_s,
                       const Eigen::Vector3d& v_s) const {
        Eigen::Vector3d d  = r_obs - r_s;
        double rho_sq      = std::max(d.squaredNorm(), 1e-9);
        double rho         = std::sqrt(rho_sq);
        double v_mag       = v_s.norm();
        double interaction = d.dot(v_s);

        double term1 = k1_ / rho;
        double term2 = k2_ * gamma_ * v_mag / rho_sq;
        double term3 = k2_ * interaction / (rho_sq * rho);

        return term1 + term2 + term3;
    }

    // ── Single link danger (trapezoid integration) ────────────────────────────
    double computeLinkDanger(const Eigen::Vector3d& r_obs,
                             const LinkState& link) const {
        double total = 0.0;
        double ds    = 1.0 / (n_pts_ - 1);

        for (int k = 0; k < n_pts_; ++k) {
            double s = static_cast<double>(k) / (n_pts_ - 1);

            Eigen::Vector3d r_s = link.r_i + s * (link.r_ip1 - link.r_i);
            Eigen::Vector3d v_s = link.v_i + s * (link.v_ip1 - link.v_i);

            double weight = (k == 0 || k == n_pts_ - 1) ? 0.5 : 1.0;
            total += weight * ds * pointDanger(r_obs, r_s, v_s);
        }
        return std::max(total, 0.0);
    }

    // ── Total danger over all links ───────────────────────────────────────────
    double computeTotalDanger(const Eigen::Vector3d& r_obs,
                              const std::vector<LinkState>& links) const {
        double total = 0.0;
        for (const auto& link : links)
            total += computeLinkDanger(r_obs, link);
        return total;
    }

    // ── Gradient at a single point s on a link ────────────────────────────────
    // This is what the nullspace controller uses per sample point
    // Returns ∇D w.r.t r_obs at position r_s
    Eigen::Vector3d computePointGradient(const Eigen::Vector3d& r_obs,
                                         const Eigen::Vector3d& r_s,
                                         const Eigen::Vector3d& v_s) const {
        Eigen::Vector3d grad = Eigen::Vector3d::Zero();
        const double eps = 1e-5;

        for (int ax = 0; ax < 3; ++ax) {
            Eigen::Vector3d r_plus  = r_obs; r_plus(ax)  += eps;
            Eigen::Vector3d r_minus = r_obs; r_minus(ax) -= eps;

            grad(ax) = (pointDanger(r_plus,  r_s, v_s) -
                        pointDanger(r_minus, r_s, v_s)) / (2.0 * eps);
        }
        return grad;
    }

    // ── Gradient over a full link (for visualisation/debugging) ──────────────
    Eigen::Vector3d computeLinkGradient(const Eigen::Vector3d& r_obs,
                                        const LinkState& link) const {
        Eigen::Vector3d grad = Eigen::Vector3d::Zero();
        const double eps = 1e-5;

        for (int ax = 0; ax < 3; ++ax) {
            Eigen::Vector3d r_plus  = r_obs; r_plus(ax)  += eps;
            Eigen::Vector3d r_minus = r_obs; r_minus(ax) -= eps;

            grad(ax) = (computeLinkDanger(r_plus,  link) -
                        computeLinkDanger(r_minus, link)) / (2.0 * eps);
        }
        return grad;
    }

    // ── Closest point on segment AB to point O ────────────────────────────────
    Eigen::Vector3d getClosestPointOnSegment(const Eigen::Vector3d& A,
                                             const Eigen::Vector3d& B,
                                             const Eigen::Vector3d& O) const {
        Eigen::Vector3d AB = B - A;
        double len2 = AB.squaredNorm();
        if (len2 < 1e-6) return A;

        double t = (O - A).dot(AB) / len2;
        t = std::max(0.0, std::min(1.0, t));
        return A + t * AB;
    }

private:
    double k1_, k2_, gamma_;
    int    n_pts_;
};