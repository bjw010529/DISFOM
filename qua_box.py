import numpy as np
import sys, os
import scipy
import time
# from sklearn.datasets import load_svmlight_file
sys.path.append(os.path.abspath("./prox_operator"))
from prox_1 import * 


def qua_loss(Sigma, xop, x, lambda_reg):
    # print(np.mean(np.dot(alpha, x) - b))
    return ((x-xop).T @ Sigma @ (x-xop)/2)[0, 0] + lambda_reg * np.sum(x**2 / (1 + x**2))

# def subop_gap(Sigma, xop, x):
#     return ((x-xop).T @ Sigma @ (x-xop)/2)[0, 0]

def qua_grad(Sigma, xop, x, lambda_reg):
    return Sigma @ (x-xop) + lambda_reg * (2 * x / (1 + x**2)**2)

def qua_batchgrad(alpha, b, x, lambda_reg):
    return alpha.T @ (alpha @ x - b) / batch_size + lambda_reg * (2 * x / (1 + x**2)**2)

# def qua_measure(Sigma, xop, x, lambda_reg):
#     return np.linalg.norm(Sigma @ (x-x_op) + lambda_reg * (2 * x / (1 + x**2)**2), np.inf)

def qua_measure(Sigma, xop, x, lambda_reg):
    g = Sigma @ (x-x_op) + lambda_reg * (2 * x / (1 + x**2)**2)
    v = g
    for i in range(np.shape(x)[0]):
        if np.abs(x[i, 0]) == R and np.sign(x[i,0]) != np.sign(g[i, 0]):
            
            v[i, 0] = 0
    return np.linalg.norm(v, np.inf)

def l2_measure(Sigma, xop, x, lambda_reg):
    return np.linalg.norm(Sigma @ (x-x_op) + lambda_reg * (2 * x / (1 + x**2)**2), 2)
# R * np.linalg.norm(Sigma @ (x-x_op), np.inf) + (x-x_op).T @ Sigma @ x

def proximal(Sigma, xop, x, lambda_reg):
    c1, beta, k = 1/4, 1/2, 1
    while True:
        a = 1
        ori = np.zeros((d, 1))
        del1 = qua_loss(Sigma, xop, x - a * qua_grad(Sigma, xop, x, lambda_reg), lambda_reg)
        del2 = qua_loss(Sigma, xop, x, lambda_reg)
        # print(x - a * qua_grad(Sigma, xop, x, lambda_reg), prox_2(x - a * qua_grad(Sigma, xop, x, lambda_reg), R))
        gradel = - a * qua_grad(Sigma, xop, x, lambda_reg)
        tmpgra = qua_grad(Sigma, xop, x, lambda_reg).T
        # print(del1, del2, tmpgra, gradel)
        while del1 - del2 > c1 * (tmpgra @ gradel)[0, 0]:
            del1 = qua_loss(Sigma, xop, x - a * qua_grad(Sigma, xop, x, lambda_reg), lambda_reg)
            del2 = qua_loss(Sigma, xop, x, lambda_reg)
            # gradel = (prox_2(x - a * qua_grad(Sigma, xop, x, lambda_reg), R) - x)
            gradel = - a * qua_grad(Sigma, xop, x, lambda_reg)
            tmpgra = qua_grad(Sigma, xop, x, lambda_reg).T
            # print(del1, del2, (tmpgra @ gradel)[0, 0])
            a *= beta
        # x_temp = prox_2(x - a * qua_grad(Sigma, xop, x, lambda_reg), R)
        x_temp = x - a * qua_grad(Sigma, xop, x, lambda_reg)
        if np.linalg.norm(x - x_temp, 1) <= 1e-10:
            # print(qua_grad(Sigma, xop, x, lambda_reg)) # test the l1-norm of the gradient
            break
        else:
            x = x_temp
            # print(np.linalg.norm(x - x_temp))
            # print(qua_loss(Sigma, xop, x, lambda_reg))
    return x, qua_loss(Sigma, xop, x, lambda_reg)

def DIFOM_minibatch(tmpsigma, xop, x0, learning_rate, M, batch_size, t=0):
    # batch_size, d = alpha.shape
    x_ret = x0
    tmp_loss = qua_loss(Sigma, xop, x_ret, lambda_reg)
    delta = tmp_loss - min_val
    print(f"Epoch {0}, Loss: {tmp_loss - min_val}")#, Gap: {subop_gap(x_ret)}
    epoch = 0
    loss_difom[0, t], fo_measure_difom[0, t], l2_difom[0, t] = delta, delta, np.linalg.norm(Sigma @ xop, np.inf)

    while True:
        # x_tilde = x_ret.copy()
        # x = x_tilde
        x = x_ret
        # alpha = np.hstack((np.random.multivariate_normal(np.zeros(d1), tmpsigma, size = batch_size), np.random.normal(size = (batch_size, d - d1))))
        # w = np.random.normal(0, 1, size = (batch_size, 1))
        alpha = np.hstack((np.clip(np.random.multivariate_normal(np.zeros(d1), np.eye(d1, d1), size = batch_size), -R, R) @ scipy.linalg.sqrtm(tmpsigma), np.clip(np.random.normal(size = (batch_size, d - d1)), -R, R)))
        w = np.clip(np.random.normal(0, 1, size = (batch_size, 1)), -R, R)
        # alpha = np.hstack((np.random.uniform(-R, R, size = (batch_size, d1)) @ scipy.linalg.sqrtm(tmpsigma), np.random.uniform(-R, R, size = (batch_size, d - d1))))
        # w = np.random.uniform(-R, R, size = (batch_size, 1))
        # w = np.zeros((batch_size, 1))
        b = np.dot(alpha, xop) + w
        grad = qua_batchgrad(alpha, b, x, lambda_reg)
        if epoch < 50:
        # grad = qua_grad(Sigma, xop, x, lambda_reg)
            time_start = time.time()
            x1 = prox_1_admm_box(x - learning_rate * grad, x, 2, R, 1, 1e-6)
            time_1 = time.time()
        x2 = prox_squared_l1_box_colvec(x - learning_rate * grad, x, -R*np.ones_like(x), R*np.ones_like(x), 2, 1e-12)
        if epoch < 50:
            time_2 = time.time()
            time_old[epoch, t], time_new[epoch, t] = time_1 - time_start, time_2 - time_1
            # print(time_1 - time_start, time_2 - time_1)
        x = x2
        if epoch >= 50:
            return loss_difom, fo_measure_difom, l2_difom
        diff[epoch, t] = np.linalg.norm(x1 - x2)
        # x = prox_1(- learning_rate * grad, 2) + x
        # print(np.shape(x))

        epoch = epoch + 1
        tmp_loss = qua_loss(Sigma, xop, x, lambda_reg)
        loss_difom[epoch, t] = (tmp_loss - min_val)# / delta
        measure, measure_2 = qua_measure(Sigma, xop, x, lambda_reg), l2_measure(Sigma, xop, x, lambda_reg)
        print(f"Epoch {epoch}, Loss: {(tmp_loss - min_val)}, Measure: {measure}")#, grad:, {(qua_grad(Sigma, xop, x, lambda_reg).T @ (x-min_x))[0,0] } # tmp_loss - min_val
        
        fo_measure_difom[epoch, t] = measure
        l2_difom[epoch, t] = measure_2
        if epoch == M:
            ret = loss_difom[np.random.randint(1, M+1), 0]
            print("loss:", ret, ", d:", d, ", delta:", delta, ", end norm:", np.linalg.norm(x_ret, 1))
            # loss[M, 0]
            # np.mean(loss[:,0], 0)
            # loss[np.random.randint(1, M+1), 0]
            return loss_difom, fo_measure_difom, l2_difom
        x_ret = x

def proximal_SGD(tmpsigma, xop, x0, learning_rate, M, batch_size, t=0):
    # batch_size, d = alpha.shape
    x_ret = x0
    tmp_loss = qua_loss(Sigma, xop, x_ret, lambda_reg)
    delta = tmp_loss - min_val
    loss_prox[0, t], fo_measure_prox[0, t], l2_prox[0, t] = delta, delta, np.linalg.norm(Sigma @ xop, np.inf)



    print(f"Epoch {0}, Loss: {tmp_loss - min_val}")#, Gap: {subop_gap(x_ret)}
    epoch = 0
    while True:
        x = x_ret
        # alpha = np.hstack((np.random.multivariate_normal(np.zeros(d1), tmpsigma, size = batch_size), np.random.normal(size = (batch_size, d - d1))))
        # w = np.random.normal(0, 1, size = (batch_size, 1))
        alpha = np.hstack((np.clip(np.random.multivariate_normal(np.zeros(d1), np.eye(d1, d1), size = batch_size), -R, R) @ scipy.linalg.sqrtm(tmpsigma), np.clip(np.random.normal(size = (batch_size, d - d1)), -R, R)))
        w = np.clip(np.random.normal(0, 1, size = (batch_size, 1)), -R, R)
        # alpha = np.hstack((np.random.uniform(-R, R, size = (batch_size, d1)) @ scipy.linalg.sqrtm(tmpsigma), np.random.uniform(-R, R, size = (batch_size, d - d1))))
        # w = np.random.uniform(-R, R, size = (batch_size, 1))
        b = np.dot(alpha, xop) + w
        grad = qua_batchgrad(alpha, b, x, lambda_reg)
        # x_temp = prox_2(x - learning_rate * grad, R)
        x_temp = prox_box(x - learning_rate * grad, R)
        # x_temp = x - learning_rate*grad

        x = x_temp
        epoch = epoch + 1
        # pre_loss = tmp_loss
        tmp_loss = qua_loss(Sigma, xop, x, lambda_reg)
        loss_prox[epoch, t] = (tmp_loss - min_val)# / delta        
        measure = qua_measure(Sigma, xop, x, lambda_reg)
        fo_measure_prox[epoch, t] = measure
        print(f"Epoch {epoch}, Loss: {(tmp_loss - min_val)}, Measure: {measure}")#, grad:, {(qua_grad(Sigma, xop, x, lambda_reg).T @ (x-min_x))[0,0] } # tmp_loss - min_val
        # if np.linalg.norm(x, np.inf) > R - 0.2:
        #     print('-------------------')
        l2_prox[epoch, t] = l2_measure(Sigma, xop, x, lambda_reg)

        if epoch == M:
            ret = loss_prox[np.random.randint(1, M+1), 0]
            print("loss:", ret, ", d:", d, ", delta:", delta, ", end norm:", np.linalg.norm(x_ret, 1))
            return loss_prox, fo_measure_prox, l2_prox
        x_ret = x


def DIFOM_svrg(tmpsigma, xop, x0, learning_rate, M, batch_size_1, t=0):
    x_ret = np.zeros(np.shape(x0))
    x = x0.copy()
    tmp_loss = qua_loss(Sigma, xop, x, lambda_reg)
    delta = tmp_loss - min_val
    epoch_length = int(np.power(batch_size_1, 1/3))
    batch_size = int(np.power(batch_size_1, 2/3))
    mm = np.floor(M/2)*(epoch_length - 1) + np.ceil(M/2)+1
    loss_difom_svrg[0, t], fo_measure_difom_svrg[0, t], l2_difom_svrg[0, t] = delta, delta, np.linalg.norm(Sigma @ xop, np.inf)

    
    print(f"Epoch {0}, Loss: {tmp_loss - min_val}")#, Gap: {subop_gap(x_ret)}
    epoch, cnt = 0, 0
    while True:
        if cnt % epoch_length == 0:
            # alpha = np.hstack((np.random.multivariate_normal(np.zeros(d1), tmpsigma, size = batch_size_1), np.random.normal(size = (batch_size_1, d - d1))))
            # w = np.random.normal(0, 1, size = (batch_size_1, 1))
            alpha = np.hstack((np.clip(np.random.multivariate_normal(np.zeros(d1), np.eye(d1, d1), size = batch_size_1), -R, R) @ scipy.linalg.sqrtm(tmpsigma), np.clip(np.random.normal(size = (batch_size_1, d - d1)), -R, R)))
            w = np.clip(np.random.normal(0, 1, size = (batch_size_1, 1)), -R, R)
            # alpha = np.hstack((np.random.uniform(-R, R, size = (batch_size_1, d1)) @ scipy.linalg.sqrtm(tmpsigma), np.random.uniform(-R, R, size = (batch_size_1, d - d1))))
            # w = np.random.uniform(-R, R, size = (batch_size_1, 1))
            b = np.dot(alpha, xop) + w

            nk_grad = qua_batchgrad(alpha, b, x, lambda_reg).reshape((-1, 1))
            x_tilde = x
            epoch += 1
            cnt += 1
            # x = prox_1(- learning_rate * nk_grad, 128) + x
            # x1 = prox_1_admm_box(x - learning_rate * nk_grad, x, 128, R, 1, epsi_hat)
            x2 = prox_squared_l1_box_colvec(x - learning_rate * nk_grad, x, -R*np.ones_like(x), R*np.ones_like(x), 128, epsi_hat)
            # print(np.linalg.norm(x1-x2))
            x = x2
            tmp_loss = qua_loss(Sigma, xop, x, lambda_reg)
            loss_difom_svrg[cnt, t] = (tmp_loss - min_val)# / delta
            measure = qua_measure(Sigma, xop, x, lambda_reg)
            fo_measure_difom_svrg[cnt, t] = measure
            l2_difom_svrg[cnt, t] = l2_measure(Sigma, xop, x, lambda_reg)

            print(f"Epoch {epoch}, Loss: {(tmp_loss - min_val)}, Measure: {measure}")#, grad:, {(qua_grad(Sigma, xop, x, lambda_reg).T @ (x-min_x))[0,0] } # tmp_loss - min_val
        else:
            cnt += 1
            if cnt % epoch_length == 0:
                epoch += 1
            # alpha = np.hstack((np.random.multivariate_normal(np.zeros(d1), tmpsigma, size = batch_size), np.random.normal(size = (batch_size, d - d1))))
            # w = np.random.normal(0, 1, size = (batch_size, 1))
            alpha = np.hstack((np.clip(np.random.multivariate_normal(np.zeros(d1), np.eye(d1, d1), size = batch_size), -R, R) @ scipy.linalg.sqrtm(tmpsigma), np.clip(np.random.normal(size = (batch_size, d - d1)), -R, R)))
            w = np.clip(np.random.normal(0, 1, size = (batch_size, 1)), -R, R)
            # alpha = np.hstack((np.random.uniform(-R, R, size = (batch_size, d1)) @ scipy.linalg.sqrtm(tmpsigma), np.random.uniform(-R, R, size = (batch_size, d - d1))))
            # w = np.random.uniform(-R, R, size = (batch_size, 1))
            b = np.dot(alpha, xop) + w

            grad = qua_batchgrad(alpha, b, x, lambda_reg).reshape((-1, 1))
            grad_nk = qua_batchgrad(alpha, b, x_tilde, lambda_reg).reshape((-1, 1))
            grad_corr = grad - grad_nk + nk_grad
            # x = prox_1(- learning_rate * grad_corr, 128) + x
            # x1 = prox_1_admm_box(x - learning_rate * grad_corr, x, 128, R, 1, epsi_hat)
            x2 = prox_squared_l1_box_colvec(x - learning_rate * grad_corr, x, -R*np.ones_like(x), R*np.ones_like(x), 128, epsi_hat)
            # print(np.linalg.norm(x1-x2))
            x = x2
            tmp_loss = qua_loss(Sigma, xop, x, lambda_reg)
            measure = qua_measure(Sigma, xop, x, lambda_reg)
            loss_difom_svrg[cnt, t] = (tmp_loss - min_val)# / delta
            fo_measure_difom_svrg[cnt, t] = measure
            l2_difom_svrg[cnt, t] = l2_measure(Sigma, xop, x, lambda_reg)
            print(f"Epoch {epoch}, Loss: {(tmp_loss - min_val)}, Measure: {measure}")#, grad:, {(qua_grad(Sigma, xop, x, lambda_reg).T @ (x-min_x))[0,0] } # tmp_loss - min_val
        # print("loss=", qua_loss(alpha, b, x, lambda_reg), "i=", epoch, "gap=", subop_gap(x))
        # if np.linalg.norm(x, np.inf) == R:
        #     print('----------------------')
        

        if epoch == M:
            ret = loss_difom_svrg[np.random.randint(1, M+1), 0]
            print("loss:", ret, ", d:", d, ", delta:", delta, ", end norm:", np.linalg.norm(x_ret, 1))
            return loss_difom_svrg, fo_measure_difom_svrg, l2_difom_svrg
        x_ret = x

def svrg(tmpsigma, xop, x0, learning_rate, M, batch_size_1, t=0):
    x_ret = np.zeros(np.shape(x0))
    x = x0.copy()
    tmp_loss = qua_loss(Sigma, xop, x, lambda_reg)
    delta = tmp_loss - min_val
    epoch_length = int(np.power(batch_size_1, 1/3))
    batch_size = int(np.power(batch_size_1, 2/3))
    mm = np.floor(M/2)*(epoch_length - 1) + np.ceil(M/2)+1
    loss_svrg[0, t], fo_measure_svrg[0, t], l2_svrg[0, t] = delta, delta, np.linalg.norm(Sigma @ xop, np.inf)

    
    print(f"Epoch {0}, Loss: {tmp_loss - min_val}")#, Gap: {subop_gap(x_ret)}
    epoch, cnt = 0, 0
    while True:
        if cnt % epoch_length == 0:
            # alpha = np.hstack((np.random.multivariate_normal(np.zeros(d1), tmpsigma, size = batch_size_1), np.random.normal(size = (batch_size_1, d - d1))))
            # w = np.random.normal(0, 1, size = (batch_size_1, 1))
            alpha = np.hstack((np.clip(np.random.multivariate_normal(np.zeros(d1), np.eye(d1, d1), size = batch_size_1), -R, R) @ scipy.linalg.sqrtm(tmpsigma), np.clip(np.random.normal(size = (batch_size_1, d - d1)), -R, R)))
            w = np.clip(np.random.normal(0, 1, size = (batch_size_1, 1)), -R, R)
            # alpha = np.hstack((np.random.uniform(-R, R, size = (batch_size_1, d1)) @ scipy.linalg.sqrtm(tmpsigma), np.random.uniform(-R, R, size = (batch_size_1, d - d1))))
            # w = np.random.uniform(-R, R, size = (batch_size_1, 1))
            b = np.dot(alpha, xop) + w

            nk_grad = qua_batchgrad(alpha, b, x, lambda_reg).reshape((-1, 1))
            x_tilde = x
            epoch += 1
            cnt += 1
            # x = prox_1(- learning_rate * nk_grad, 128) + x
            x = prox_box(x - learning_rate * nk_grad, R)
            tmp_loss = qua_loss(Sigma, xop, x, lambda_reg)
            loss_svrg[cnt, t] = (tmp_loss - min_val)# / delta
            measure = qua_measure(Sigma, xop, x, lambda_reg)
            fo_measure_svrg[cnt, t] = measure
            if epoch <= 300:
                l2_svrg[cnt, t] = l2_measure(Sigma, xop, x, lambda_reg)

            print(f"Epoch {epoch}, Loss: {(tmp_loss - min_val)}, Measure: {measure}")#, grad:, {(qua_grad(Sigma, xop, x, lambda_reg).T @ (x-min_x))[0,0] } # tmp_loss - min_val
        else:
            cnt += 1
            if cnt % epoch_length == 0:
                epoch += 1
            # alpha = np.hstack((np.random.multivariate_normal(np.zeros(d1), tmpsigma, size = batch_size), np.random.normal(size = (batch_size, d - d1))))
            # w = np.random.normal(0, 1, size = (batch_size, 1))
            alpha = np.hstack((np.clip(np.random.multivariate_normal(np.zeros(d1), np.eye(d1, d1), size = batch_size), -R, R) @ scipy.linalg.sqrtm(tmpsigma), np.clip(np.random.normal(size = (batch_size, d - d1)), -R, R)))
            w = np.clip(np.random.normal(0, 1, size = (batch_size, 1)), -R, R)
            # alpha = np.hstack((np.random.uniform(-R, R, size = (batch_size, d1)) @ scipy.linalg.sqrtm(tmpsigma), np.random.uniform(-R, R, size = (batch_size, d - d1))))
            # w = np.random.uniform(-R, R, size = (batch_size, 1))
            b = np.dot(alpha, xop) + w

            grad = qua_batchgrad(alpha, b, x, lambda_reg).reshape((-1, 1))
            grad_nk = qua_batchgrad(alpha, b, x_tilde, lambda_reg).reshape((-1, 1))
            grad_corr = grad - grad_nk + nk_grad
            # x = prox_1(- learning_rate * grad_corr, 128) + x
            x = prox_box(x - learning_rate * grad_corr, R)
            tmp_loss = qua_loss(Sigma, xop, x, lambda_reg)
            measure = qua_measure(Sigma, xop, x, lambda_reg)
            loss_svrg[cnt, t] = (tmp_loss - min_val)# / delta
            fo_measure_svrg[cnt, t] = measure
            if epoch <= 300:
                l2_svrg[cnt, t] = l2_measure(Sigma, xop, x, lambda_reg)
            print(f"Epoch {epoch}, Loss: {(tmp_loss - min_val)}, Measure: {measure}")
        # if np.linalg.norm(x, np.inf) == R:
        #     print('----------------------')
        

        if epoch == M:
            ret = loss_svrg[np.random.randint(1, M+1), 0]
            print("loss:", ret, ", d:", d, ", delta:", delta, ", end norm:", np.linalg.norm(x_ret, 1))
            return loss_svrg, fo_measure_svrg, l2_svrg
        x_ret = x

def SMD_minibatch(tmpsigma, xop, x0, learning_rate, M, batch_size, t=0):
    x_ret = x0
    tmp_loss = qua_loss(Sigma, xop, x_ret, lambda_reg)
    # pre_loss = 0
    delta = tmp_loss - min_val
    loss_smd[0, t], fo_measure_smd[0, t], l2_smd[0, t] = delta, delta, np.linalg.norm(Sigma @ xop, np.inf)
    rho = lambda_reg / 2 - 1
    c = np.sqrt((tmp_loss)/(rho * L**2))
    alpha_k = c / np.sqrt(M)

    print(f"Epoch {0}, Loss: {tmp_loss - min_val}")#, Gap: {subop_gap(x_ret)}
    epoch = 0
    while True:
        # x_tilde = x_ret.copy()
        # x = x_tilde
        x = x_ret
        # alpha = np.hstack((np.random.multivariate_normal(np.zeros(d1), tmpsigma, size = batch_size), np.random.normal(size = (batch_size, d - d1))))
        # w = np.random.normal(0, 1, size = (batch_size, 1))
        alpha = np.hstack((np.clip(np.random.multivariate_normal(np.zeros(d1), np.eye(d1, d1), size = batch_size), -R, R) @ scipy.linalg.sqrtm(tmpsigma), np.clip(np.random.normal(size = (batch_size, d - d1)), -R, R)))
        w = np.clip(np.random.normal(0, 1, size = (batch_size, 1)), -R, R)
        # alpha = np.hstack((np.random.uniform(-R, R, size = (batch_size, d1)) @ scipy.linalg.sqrtm(tmpsigma), np.random.uniform(-R, R, size = (batch_size, d - d1))))
        # w = np.random.uniform(-R, R, size = (batch_size, 1))
        # w = np.zeros((batch_size, 1))
        b = np.dot(alpha, xop) + w
        grad = qua_batchgrad(alpha, b, x, lambda_reg)
        # x_temp = prox_MD(grad, x, alpha_k, 1+1/np.log(np.size(x)))
        x_temp_2 = prox_md_box(x, grad, alpha_k, -R*np.ones_like(x), R*np.ones_like(x), p=1+1/np.log(np.size(x)))
        # print("diff: ", np.linalg.norm(x_temp - x_temp_2), np.linalg.norm(x_temp), np.linalg.norm(x_temp_2))
        x = x_temp_2
        epoch = epoch + 1
        # pre_loss = tmp_loss
        tmp_loss = qua_loss(Sigma, xop, x, lambda_reg)
        loss_smd[epoch, t] = (tmp_loss - min_val)# / delta        
        measure = qua_measure(Sigma, xop, x, lambda_reg)
        fo_measure_smd[epoch, t] = measure
        l2_smd[epoch, t] = l2_measure(Sigma, xop, x, lambda_reg)
        print(f"Epoch {epoch}, Loss: {(tmp_loss - min_val)}, Measure: {measure}")#, grad:, {(qua_grad(Sigma, xop, x, lambda_reg).T @ (x-min_x))[0,0] } # tmp_loss - min_val
        # if np.linalg.norm(x, np.inf) == R:
        #     print('----------------------')
        if epoch == M:
            ret = loss_smd[np.random.randint(1, M+1), 0]
            # loss[M, 1]
            # np.mean(loss[:,1], 0)
            # loss[np.random.randint(1, M+1), 0]
            print("loss:", ret, ", d:", d, ", delta:", delta, ", end norm:", np.linalg.norm(x_ret, 1))
            return loss_smd, fo_measure_smd, l2_smd
        x_ret = x

def SMD_svrg(tmpsigma, xop, x0, learning_rate, M, batch_size_1, t=0):
    x_ret = np.zeros(np.shape(x0))
    x = x0.copy()
    tmp_loss = qua_loss(Sigma, xop, x, lambda_reg)
    delta = tmp_loss - min_val
    epoch_length = int(np.power(batch_size_1, 1/3))
    batch_size = int(np.power(batch_size_1, 2/3))
    mm = np.floor(M/2)*(epoch_length - 1) + np.ceil(M/2)+1
    loss_smd_svrg[0, t], fo_measure_smd_svrg[0, t], l2_smd_svrg[0, t] = delta, delta, np.linalg.norm(Sigma @ xop, np.inf)
    rho = lambda_reg / 2 - 1
    c = np.sqrt((tmp_loss)/(rho * L**2))
    alpha_k = c / np.sqrt(M)

    
    print(f"Epoch {0}, Loss: {tmp_loss - min_val}")
    epoch, cnt = 0, 0
    while True:
        if cnt % epoch_length == 0:
            # alpha = np.hstack((np.random.multivariate_normal(np.zeros(d1), tmpsigma, size = batch_size_1), np.random.normal(size = (batch_size_1, d - d1))))
            # w = np.random.normal(0, 1, size = (batch_size_1, 1))
            alpha = np.hstack((np.clip(np.random.multivariate_normal(np.zeros(d1), np.eye(d1, d1), size = batch_size_1), -R, R) @ scipy.linalg.sqrtm(tmpsigma), np.clip(np.random.normal(size = (batch_size_1, d - d1)), -R, R)))
            w = np.clip(np.random.normal(0, 1, size = (batch_size_1, 1)), -R, R)
            # alpha = np.hstack((np.random.uniform(-R, R, size = (batch_size_1, d1)) @ scipy.linalg.sqrtm(tmpsigma), np.random.uniform(-R, R, size = (batch_size_1, d - d1))))
            # w = np.random.uniform(-R, R, size = (batch_size_1, 1))
            b = np.dot(alpha, xop) + w

            nk_grad = qua_batchgrad(alpha, b, x, lambda_reg).reshape((-1, 1))
            x_tilde = x
            epoch += 1
            cnt += 1
            # x = prox_1(- learning_rate * nk_grad, 128) + x
            # x1 = prox_MD(nk_grad, x, alpha_k, 1+1/np.log(np.size(x)))
            x2 = prox_md_box(x, nk_grad, alpha_k, -R*np.ones_like(x), R*np.ones_like(x), p=1+1/np.log(np.size(x)))
            # print(np.linalg.norm(x1-x2))
            x = x2
            tmp_loss = qua_loss(Sigma, xop, x, lambda_reg)
            loss_smd_svrg[cnt, t] = (tmp_loss - min_val)# / delta
            measure = qua_measure(Sigma, xop, x, lambda_reg)
            fo_measure_smd_svrg[cnt, t] = measure
            l2_smd_svrg[cnt, t] = l2_measure(Sigma, xop, x, lambda_reg)

            print(f"Epoch {epoch}, Loss: {(tmp_loss - min_val)}, Measure: {measure}")#, grad:, {(qua_grad(Sigma, xop, x, lambda_reg).T @ (x-min_x))[0,0] } # tmp_loss - min_val
        else:
            cnt += 1
            if cnt % epoch_length == 0:
                epoch += 1
            # alpha = np.hstack((np.random.multivariate_normal(np.zeros(d1), tmpsigma, size = batch_size), np.random.normal(size = (batch_size, d - d1))))
            # w = np.random.normal(0, 1, size = (batch_size, 1))
            alpha = np.hstack((np.clip(np.random.multivariate_normal(np.zeros(d1), np.eye(d1, d1), size = batch_size), -R, R) @ scipy.linalg.sqrtm(tmpsigma), np.clip(np.random.normal(size = (batch_size, d - d1)), -R, R)))
            w = np.clip(np.random.normal(0, 1, size = (batch_size, 1)), -R, R)
            # alpha = np.hstack((np.random.uniform(-R, R, size = (batch_size, d1)) @ scipy.linalg.sqrtm(tmpsigma), np.random.uniform(-R, R, size = (batch_size, d - d1))))
            # w = np.random.uniform(-R, R, size = (batch_size, 1))
            b = np.dot(alpha, xop) + w

            grad = qua_batchgrad(alpha, b, x, lambda_reg).reshape((-1, 1))
            grad_nk = qua_batchgrad(alpha, b, x_tilde, lambda_reg).reshape((-1, 1))
            grad_corr = grad - grad_nk + nk_grad
            # x1 = prox_MD(grad_corr, x, alpha_k, 1+1/np.log(np.size(x)))
            x2 = prox_md_box(x, grad_corr, alpha_k, -R*np.ones_like(x), R*np.ones_like(x), C=1+1/np.log(np.size(x)))
            # print(np.linalg.norm(x1-x2))
            x = x2
            tmp_loss = qua_loss(Sigma, xop, x, lambda_reg)
            measure = qua_measure(Sigma, xop, x, lambda_reg)
            loss_smd_svrg[cnt, t] = (tmp_loss - min_val)# / delta
            fo_measure_smd_svrg[cnt, t] = measure
            l2_smd_svrg[cnt, t] = l2_measure(Sigma, xop, x, lambda_reg)
            print(f"Epoch {epoch}, Loss: {(tmp_loss - min_val)}, Measure: {measure}")#, grad:, {(qua_grad(Sigma, xop, x, lambda_reg).T @ (x-min_x))[0,0] } # tmp_loss - min_val
        # if np.linalg.norm(x, np.inf) == R:
        #     print('----------------------')
        

        if epoch == M:
            ret = loss_smd_svrg[np.random.randint(1, M+1), 0]
            print("loss:", ret, ", d:", d, ", delta:", delta, ", end norm:", np.linalg.norm(x_ret, 1))
            return loss_smd_svrg, fo_measure_smd_svrg, l2_smd_svrg
        x_ret = x



np.random.seed(15)
c = 7
R = 3
d, d1 = 10000, 100#
d = int(sys.argv[1])
print(d)
# d1 = int(d / 16)
d1 = 100
# epsi_hat = epsi / (R * L)
epsi_hat = 1e-10
num_exp = 3
lambda_reg = 2.5

# if not os.path.exists("./data/save_para/save" + str(d) + "_1.npz") or not os.path.exists("./data/save_para/save" + str(d) + "_2.npz") or d == d1:
Sigma = scipy.sparse.eye(d, format = 'csr')
Q = scipy.linalg.orth(np.random.uniform(0, 1, size = (d1, d1)))
D = np.diag(np.random.uniform(1, 2, size = d1))
tmpsigma = np.dot(np.dot(Q, D), Q.T)
i, j, v, cnt = np.zeros(d1*d1), np.zeros(d1*d1), np.zeros(d1*d1), 0
for ii in range(d1):
    for jj in range(d1):
        i[cnt], j[cnt], v[cnt] = ii, jj, tmpsigma[ii,jj]
        if ii == jj:
            v[cnt] -= 1
        cnt += 1
# Sigma = Sigma + scipy.sparse.csr_matrix((v, (i,j)), shape = (d, d))
Sigma = (Sigma + scipy.sparse.csr_matrix((v, (i,j)), shape = (d, d))) * (1 - 2*R * np.exp(-R**2/2)/(np.sqrt(2*np.pi) * 0.99730020393674))
# Sigma = scipy.sparse.csr_matrix((v, (i,j)), shape = (d, d))


# x1, x2, r = np.random.uniform(-1, 1, size = (d, 1)), np.zeros((d, 1)), np.random.uniform()
# x2[np.random.choice(range(d), 3, replace = False), 0] = np.random.uniform(-1, 1, 3)
# x_op = r * x1 / np.linalg.norm(x1) + (1 - r) * x2 / np.linalg.norm(x2)
x_op = np.zeros((d, 1))
# print(np.shape(np.ones((d1, 1))), np.shape(x_op[range(d1), :]))
x_op[range(d1), :] = np.ones((d1, 1)) #np.random.uniform(-1, 1, d1)
# x_op = np.ones((d, 1))
# x_op = x_op / np.linalg.norm(x_op)
# print(x_op)
# scipy.sparse.save_npz("./data/save_para/save" + str(d) + "_1.npz", Sigma)
# np.savez("./data/save_para/save" + str(d) + "_2.npz", x_op, tmpsigma)
# else:
#     Sigma = scipy.sparse.load_npz("./data/save_para/save" + str(d) + "_1.npz")
#     x_op, tmpsigma = np.load("./data/save_para/save" + str(d) + "_2.npz")['arr_0'], np.load("./data/save_para/save" + str(d) + "_2.npz")['arr_1']


sigma = 2 * np.max(Sigma.diagonal())
vals, vecs = scipy.sparse.linalg.eigs(Sigma, which = 'LR')
# print(type(np.max(np.real(vals))), type(lambda_reg), type(x_op), type(tmpsigma), tmpsigma)
L = np.max(np.real(vals)) + 2 * lambda_reg
learning_rate = 1 / L


epsi = 3


x0 = np.zeros((d, 1))
# Delta = qua_loss(alpha, b, x0, lambda_reg)
# batch_size = int(c * R ** 2 * np.log(d) * (sigma ** 2) / (epsi ** 2))
# batch_size = 2500 or 1000
batch_size = 1000

min_x, min_val = proximal(Sigma, x_op, x_op, lambda_reg)
# print(min_val)
# min_x, min_val = proximal(Sigma, x_op, x_op, lambda_reg)
# print(min_val)
Delta = qua_loss(Sigma, x_op, x0, lambda_reg) - min_val
# M = int(200 * (R ** 2) / 3)
M = 300
# M = int(Delta * R ** 2 * L / (epsi ** 2))
mm = np.floor(M/2)*(int(np.power(batch_size, 1/3)) - 1) + np.ceil(M/2)+1
print(batch_size,M)

loss_difom, fo_measure_difom, l2_difom = np.zeros((M+1, num_exp)), np.zeros((M+1, num_exp)), np.zeros((M+1, num_exp))
loss_prox, fo_measure_prox, l2_prox = np.zeros((M+1, num_exp)), np.zeros((M+1, num_exp)), np.zeros((M+1, num_exp))
loss_smd, fo_measure_smd, l2_smd = np.zeros((M+1, num_exp)), np.zeros((M+1, num_exp)), np.zeros((M+1, num_exp))
loss_svrg, fo_measure_svrg, l2_svrg = np.zeros((int(mm), num_exp)), np.zeros((int(mm), num_exp)), np.zeros((int(mm), num_exp))
loss_difom_svrg, fo_measure_difom_svrg, l2_difom_svrg = np.zeros((int(mm), num_exp)), np.zeros((int(mm), num_exp)), np.zeros((int(mm), num_exp))
loss_smd_svrg, fo_measure_smd_svrg, l2_smd_svrg = np.zeros((int(mm), num_exp)), np.zeros((int(mm), num_exp)), np.zeros((int(mm), num_exp))
time_old, time_new, diff = np.zeros((50, num_exp)), np.zeros((50, num_exp)), np.zeros((50, num_exp))
print(np.linalg.norm(min_x), min_val)
# if not os.path.exists("./data/save_para/loss" + str(d) + "_difom.csv") or not os.path.exists("./data/save_para/loss" + str(d) + "_prox.csv") or not os.path.exists("./data/save_para/loss" + str(d) + "_difom_svrg.csv"):
for t in range(num_exp):
    loss_difom, fo_measure_difom, l2_difom = DIFOM_minibatch(tmpsigma, x_op, x0, learning_rate, M, batch_size, t)
    # loss_prox, fo_measure_prox, l2_prox = proximal_SGD(tmpsigma, x_op, x0, learning_rate, M, batch_size, t)
    # loss_smd, fo_measure_smd, l2_smd = SMD_minibatch(tmpsigma, x_op, x0, learning_rate, M, batch_size, t)
    # loss_svrg, fo_measure_svrg, l2_svrg = svrg(tmpsigma, x_op, x0, learning_rate/10, M, batch_size, t)
    # loss_difom_svrg, fo_measure_difom_svrg, l2_difom_svrg = DIFOM_svrg(tmpsigma, x_op, x0, learning_rate, M, batch_size, t)
    # loss_smd_svrg, fo_measure_smd_svrg, l2_smd_svrg = SMD_svrg(tmpsigma, x_op, x0, learning_rate, M, batch_size, t)
# np.savetxt("./data/save_para/loss" + str(d) + "_difom.csv",loss_difom)
# np.savetxt("./data/save_para/fo_measure" + str(d) + "_difom.csv",fo_measure_difom)
# np.savetxt("./data/save_para/l2" + str(d) + "_difom.csv",l2_difom)
np.savetxt("./data/save_para/time_old" + str(d) + ".csv", time_old)
np.savetxt("./data/save_para/time_new" + str(d) + ".csv", time_new)
np.savetxt("./data/save_para/diff" + str(d) + ".csv", diff)
# np.savetxt("./data/save_para/loss" + str(d) + "_prox.csv",loss_prox)
# np.savetxt("./data/save_para/fo_measure" + str(d) + "_prox.csv",fo_measure_prox)
# np.savetxt("./data/save_para/l2" + str(d) + "_prox.csv",l2_prox)
# np.savetxt("./data/save_para/loss" + str(d) + "_smd.csv",loss_smd)
# np.savetxt("./data/save_para/fo_measure" + str(d) + "_smd.csv",fo_measure_smd)
# np.savetxt("./data/save_para/l2" + str(d) + "_smd.csv",l2_smd)

# np.savetxt("./data/save_para/loss" + str(d) + "_svrg.csv",loss_svrg)
# np.savetxt("./data/save_para/fo_measure" + str(d) + "_svrg.csv",fo_measure_svrg)
# np.savetxt("./data/save_para/l2" + str(d) + "_svrg.csv",l2_svrg)
# np.savetxt("./data/save_para/loss" + str(d) + "_difom_svrg.csv",loss_difom_svrg)
# np.savetxt("./data/save_para/fo_measure" + str(d) + "_difom_svrg.csv",fo_measure_difom_svrg)
# np.savetxt("./data/save_para/l2" + str(d) + "_difom_svrg.csv",l2_difom_svrg)
# np.savetxt("./data/save_para/loss" + str(d) + "_smd_svrg.csv",loss_smd_svrg)
# np.savetxt("./data/save_para/fo_measure" + str(d) + "_smd_svrg.csv",fo_measure_smd_svrg)
# np.savetxt("./data/save_para/l2" + str(d) + "_smd_svrg.csv",l2_smd_svrg)


'''

'''


'''

'''





