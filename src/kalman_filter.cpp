#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
    x_ = x_in;
    P_ = P_in;
    F_ = F_in;
    H_ = H_in;
    R_ = R_in;
    Q_ = Q_in;
}

void KalmanFilter::Predict() {
    x_ = F_ * x_; // There is no external motion, so, we do not have to add "+u"
    MatrixXd Ft = F_.transpose();
    P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
    VectorXd z_pred = H_ * x_;
    VectorXd y = z - z_pred;
    MatrixXd Ht = H_.transpose();
    MatrixXd S = H_ * P_ * Ht + R_;
    MatrixXd Si = S.inverse();
    MatrixXd PHt = P_ * Ht;
    MatrixXd K = PHt * Si;

    //new estimate
    estimate(y, K);
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
    double px = x_[0];
    double py = x_[1];
    double vx = x_[2];
    double vy = x_[3];
    double rho = sqrt(px * px + py * py);
    double phi = atan2(py, px);
    double rho_dot = (px * vx + py * vy) / rho;

    VectorXd z_pred(3);
    z_pred << rho, phi, rho_dot;
    VectorXd y = z - z_pred;
    // resulting angle phi in the y vector should be adjusted so that it is between -pi and pi
    double PI = atan(1)*4;
    if( y[1] > PI ) y[1] -= 2*PI;
    if( y[1] < -PI ) y[1] += 2*PI;

    MatrixXd Ht = H_.transpose();
    MatrixXd S = H_ * P_ * Ht + R_;
    MatrixXd Si = S.inverse();
    MatrixXd PHt = P_ * Ht;
    MatrixXd K = PHt * Si;

    //new estimate
    estimate(y, K);
}

void KalmanFilter::estimate(const VectorXd &y, const MatrixXd &K) {
    x_ = x_ + (K * y);
    long x_size = x_.size();
    MatrixXd I = MatrixXd::Identity(x_size, x_size);
    P_ = (I - K * H_) * P_;
}
