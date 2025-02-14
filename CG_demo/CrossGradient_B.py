import numpy as np
from scipy.sparse import diags, kron, eye, csr_matrix, vstack
from scipy.sparse.linalg import spsolve
from scipy.ndimage import gaussian_filter
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.sparse import issparse
from scipy.sparse import coo_matrix
from scipy.fft import fft2, ifft2, fftshift, ifftshift




class CrossGradient:
    def __init__(self):
        pass

    def dx_dz(self, SA, SB, a_b='ab',ni = 50,tol_ = 1e-20, smooth=False,coef_x=2,coef_y=2):
        CG_W = SA.copy()
        CG_R = SB.copy()
        
        nz, nx = CG_W.shape
        Dx, Dz = self._Dx_Dz(nz, nx)
        
        CGx_WR = self._cross2d(CG_W, CG_R)
        JR = self._crossJa(CG_R)
        # npp=JR.toarray()
        JW = self._crossJb(CG_W)
        # npp=JW.toarray()
        J = JW + JR
        
        ga = JW @ CGx_WR.ravel(order='F')
        ga = self._normali(ga)
        ga = ga.reshape(CG_W.T.shape).T
        
        gb = JR @ CGx_WR.ravel(order='F')
        gb = self._normali(gb)
        gb = gb.reshape(CG_R.T.shape).T
        
        a_ = CG_W.copy()
        b_ = CG_R.copy()
        ka = np.linspace(-100, 100, 50)
        kb = np.linspace(-100, 100, 50)
        tol_ = 1e-20
        da_ = np.zeros_like(CG_W)
        db_ = np.zeros_like(CG_R)
        
        if a_b == 'ab':
            a_, b_, _, _ = self._cross_ab(CG_W, CG_R, ni, tol_, ka, kb, Dx, Dz, 2, 2)
            if smooth:
                b_ = self._smooth2a(b_, coef_x, coef_y)
                a_ = self._smooth2a(a_, coef_x, coef_y)
            
        elif a_b == 'a':
            a_, _ = self._cross_a(CG_W, CG_R, ni, tol_, ka, Dx, Dz, 2, 2)
            if smooth:
                a_ = self._smooth2a(a_, coef_x, coef_y)           
                
        elif a_b == 'b':
            b_, _ = self._cross_b(CG_W, CG_R, ni, tol_, kb, Dx, Dz, 2, 2)
            if smooth:
                b_ = self._smooth2a(b_, coef_x, coef_y)
                
        elif a_b == 'No':
            a_ = CG_W
            b_ = CG_R
        
        SA_CG = a_
        SB_CG = b_
        return SA_CG, SB_CG

    def _Dx_Dz(self, nz, nx):
        Dx = self._DX(nz, nx)
        # npp=Dx.toarray()
        Dz = self._DZ(nz, nx)
        return Dz, Dx 

    def _DZ(self, nz, nx):
        # Forward differences
        fwd = [-1.5]*nz + [2.0]*nz + [-0.5]*nz + [0.0]*(nz*nx - 3*nz)
        rows = np.tile(np.arange(nz), nx)
        cols = np.arange(nz*nx)
        dz_fwd = csr_matrix((fwd, (rows, cols)), shape=(nz, nz*nx))
        # npp=dz_fwd.toarray()
        # Central differences
        ctd = [-0.5]*(nz*nx - 2*nz) + [0.0]*(nz*nx - 2*nz) + [0.5]*(nz*nx - 2*nz)
        I = np.tile(np.arange(nz*nx - 2*nz), 3)
        J = np.concatenate([
            np.arange(nz*nx - 2*nz),
            np.arange(nz, nz*nx - 2*nz + nz),
            np.arange(2*nz, nz*nx - 2*nz + 2*nz)
        ])
        dz_ctd = csr_matrix((ctd, (I, J)), shape=(nz*nx - 2*nz, nz*nx))
        
        # Backward differences
        bwd = [0.0]*(nz*nx - 3*nz) + [0.5]*nz + [-2.0]*nz + [1.5]*nz
        dz_bwd = csr_matrix((bwd, (rows, cols)), shape=(nz, nz*nx))
        
        # Combine
        Dz = vstack([dz_fwd, dz_ctd, dz_bwd])
        # npp=Dz.toarray()
        return Dz

    def _DX(self, nz, nx):
        Dx_1d = self._D_x(nz)
        Id = eye(nx)
        Dx = kron(Id, Dx_1d)
        # npp=Dx.toarray()
        return Dx

    def _D_x(self, n):
        fwd = [-1.5, 2.0, -0.5] + [0.0]*(n-3)
        # rows = [0]*(n-3) + [1]*(n-2) + [2]*(n-3)
        # cols = np.arange(n)
        dx_fwd = csr_matrix(fwd, shape=(1, n))
        # npp=dx_fwd.toarray()
        ctd = [-0.5]*(n-2) + [0.0]*(n-2) + [0.5]*(n-2)
        I = np.tile(np.arange(n-2), 3)
        J = np.concatenate([
            np.arange(n-2),
            np.arange(1, n-1),
            np.arange(2, n)
        ])
        dx_ctd = csr_matrix((ctd, (I, J)), shape=(n-2, n))
        
        bwd = [0.0]*(n-3) + [0.5, -2.0, 1.5]
        dx_bwd = csr_matrix(bwd, shape=(1, n))
        # npp=dx_bwd.toarray()
        Dx_ = vstack([dx_fwd, dx_ctd, dx_bwd])
        return Dx_


    def _cross2d(self, a, b, Dx=None, Dz=None):
        """
        计算两个矩阵的二维交叉积（行列式形式）
        参数：
            a, b: 二维numpy数组，需同尺寸
            Dx, Dz: 可选的稀疏微分算子矩阵
        返回：
            xab: 与a同形的二维数组
        """
        # 输入检查
        assert a.shape == b.shape, "Matrices a and b must have the same shape"
        
        # 自动生成微分算子（如果未提供）
        if Dx is None or Dz is None:
            nz, nx = a.shape
            Dx, Dz = self._Dx_Dz(nz, nx)
        
        # 展平矩阵并进行微分运算
        a_flat = a.ravel(order='F')  # 按列展开（MATLAB风格）
        b_flat = b.ravel(order='F')
        
        # 确保使用稀疏矩阵乘法
        if issparse(Dx):
            Dx_a = Dx @ a_flat
            Dz_b = Dz @ b_flat
            Dz_a = Dz @ a_flat
            Dx_b = Dx @ b_flat
        else:
            Dx_a = Dx.dot(a_flat)
            Dz_b = Dz.dot(b_flat)
            Dz_a = Dz.dot(a_flat)
            Dx_b = Dx.dot(b_flat)
        
        # 计算交叉积（行列式）
        xab = np.multiply(Dx_a, Dz_b) - np.multiply(Dz_a, Dx_b)
        
        # 重塑为原始形状（保持MATLAB的列优先顺序）
        xab =  xab.reshape(a.shape, order='F')
        
        # 可选归一化（根据原始MATLAB代码）
        # xab = self._normali(xab)
        
        return  xab
    

    def _crossJa(self, b, Dx=None, Dz=None):
        
        if Dx is None or Dz is None:
            nz, nx = b.shape
            Dx, Dz = self._Dx_Dz(nz, nx)
            
        b_flat = b.ravel(order='F')
        
        Dz_b=(Dz @ b_flat).reshape(-1,1)
        Dx_b=(Dx @ b_flat).reshape(-1,1)
        
        
        term1 = self._sparse_full(Dx, Dz_b)
        term2 = self._sparse_full(Dz, Dx_b)
        Ja = (term1-term2).T
        # npp=Ja.toarray()
        return Ja



    def _sparse_full(self, A, b):
        """
    精确对应MATLAB的sparse_full函数
    :param A: 稀疏矩阵 (任何scipy.sparse格式)
    :param b: numpy数组 (列向量形式)
    :return: 新的稀疏矩阵 (CSR格式)
        """
    # 转换为COO格式获取坐标和值
        A_coo = A.tocoo()
        rows = A_coo.row    # MATLAB的i
        cols = A_coo.col    # MATLAB的j
        data = A_coo.data
    
    # 转换为MATLAB的线性索引（列优先）
    # sub2ind(size(A),i,j)的等效实现
        matlab_linear_indices = rows + cols * A.shape[0]  # 关键！列优先计算
    
    # 处理MATLAB的索引自动循环特性
    # 当索引超过b长度时循环取值
        b = b.flatten()  # 确保b是向量
        b_indices = matlab_linear_indices % len(b)
    
    # 执行元素乘法（完全模拟MATLAB行为）
        new_data = data * b[b_indices]
    
    # 重建稀疏矩阵（保持原始行列坐标）
        return coo_matrix((new_data, (rows, cols)), shape=A.shape)




    def _crossJb(self, a, Dx=None, Dz=None):
        
        if Dx is None or Dz is None:
            nz, nx = a.shape
            Dx, Dz = self._Dx_Dz(nz, nx)
        Jb = -self._crossJa(a, Dx, Dz)
        
        return Jb




    def _normali(self, a):
        a_max = np.max(np.abs(a))
        return a / a_max if a_max != 0 else a


    def _smooth2a(self, matrix, Nr, Nc):
        # 简化的二维平滑实现
        from scipy.signal import convolve2d
        kernel = np.ones((2*Nr+1, 2*Nc+1)) / ((2*Nr+1)*(2*Nc+1))
        return convolve2d(matrix, kernel, mode='same', boundary='symm')

    def _image_gaussian(self,image_xy, a, b, filt_TYPE):
        """
    高斯滤波图像处理（低通/高通）
    
    参数：
        image_xy : numpy.ndarray, 输入的二维图像矩阵
        a, b     : float, 高斯滤波参数（控制滤波器宽度）
        filt_TYPE: str, 滤波类型，'LOW_PASS'（低通）或 'HI_PASS'（高通）
    
    返回：
        xy_image : numpy.ndarray, 滤波后的图像（空间域）
        image_kk : numpy.ndarray, 频域的图像（中心化）
        kk_filter: numpy.ndarray, 生成的滤波器矩阵
        """
    # 确保输入是二维矩阵
        assert image_xy.ndim == 2, "输入必须是二维矩阵"

    # --------------------------------------------------------------------------
    # Step 1: 零填充至2的幂次（优化FFT性能）
    # --------------------------------------------------------------------------
        m, n = image_xy.shape
        m_ = 2 ** int(np.ceil(np.log2(m)))   # 计算最近的2的幂次
        n_ = 2 ** int(np.ceil(np.log2(n)))
        m_extra = m_ - m
        n_extra = n_ - n

    # 向右和下侧填充零（MATLAB默认填充方式）
        image_pad = np.pad(image_xy, 
                           ((0, m_extra), (0, n_extra)), 
                           mode='constant')

    # --------------------------------------------------------------------------
    # Step 2: 傅里叶变换到频域
    # --------------------------------------------------------------------------
        k_image_k = fft2(image_pad)

    # --------------------------------------------------------------------------
    # Step 3: 构建高斯滤波器
    # --------------------------------------------------------------------------
    # 调整滤波器参数（与MATLAB一致）
        a_adj = n_ * (a / 2) * np.sqrt(2)
        b_adj = m_ * (b / 2) * np.sqrt(2)

    # 生成网格坐标（MATLAB的meshgrid默认是'xy'模式）
        X = np.arange(n_)
        Y = np.arange(m_)
        XY, YX = np.meshgrid(X, Y)  # XY对应列方向，YX对应行方向

    # 计算中心点（MATLAB索引从1开始，Python从0开始，但这里用n_/2直接对应）
        x_center = n_ / 2
        y_center = m_ / 2

    # 计算高斯分布
        kk_filter = ((XY - x_center) / a_adj)**2 + ((YX - y_center) / b_adj)**2
        kk_filter = -kk_filter

    # 根据滤波类型处理
        if filt_TYPE == 'LOW_PASS':
            kk_filter = np.exp(kk_filter)
        elif filt_TYPE == 'HI_PASS':
            kk_filter = 1 - np.exp(kk_filter)
        else:
            raise ValueError("filt_TYPE 必须是 'LOW_PASS' 或 'HI_PASS'")

    # 归一化滤波器
        max_val = np.max(np.abs(kk_filter))
        if max_val != 0:
            kk_filter /= max_val

    # --------------------------------------------------------------------------
    # Step 4: 频域滤波
    # --------------------------------------------------------------------------
    # 移动滤波器到FFT对应的角频率位置
        k_filter_k = ifftshift(kk_filter)  # 对应MATLAB的fftshift

    # 频域相乘（注意：Python的fft2结果已经是正确的象限顺序）
        filtered_k = k_filter_k * k_image_k

    # --------------------------------------------------------------------------
    # Step 5: 逆变换回空间域
    # --------------------------------------------------------------------------
        xy_image = ifft2(filtered_k).real  # 取实数部分

    # 裁剪填充的零
        xy_image = xy_image[:m, :n]

    # --------------------------------------------------------------------------
    # 记录中间结果
    # --------------------------------------------------------------------------
        image_kk = fftshift(k_image_k)  # 中心化的频域图像

        return xy_image, image_kk, kk_filter
    
    
    
    def _cross_ab(self, a, b, ni, tol_, ka, kb, Dx, Dz, ax=None, az=None):
        no_print = False
        step_a = 1.0
        step_b = 1.0
        da_ = np.zeros_like(a)
        db_ = np.zeros_like(b)
        E_ = np.inf
        i_ = 0
    
        while E_ > tol_ and i_ < ni:
        # ------------------------- a update -------------------------
        # Forward calculation
            xab_ = self._cross2d(a, b, Dx, Dz)
        
        # Gradient calculation
            Ja = self._crossJa(b, Dx, Dz)
            # npp=Ja.toarray()
            ga = Ja @ xab_.ravel(order='F')
        
        # Hessian and filtering (placeholder)
            if ax is not None and az is not None:
                ga = ga.reshape(a.T.shape).T
                ga = self._image_gaussian(ga, ax, az, 'LOW_PASS')  # To be implemented
                ga=ga[0]
                ga = ga.ravel(order='F')
            ga = self._normali(ga)
        
        # Solve linear system
            A = Ja @ Ja.T + step_a * eye(a.size, format='csr')
            ga = spsolve(A, ga).reshape(a.T.shape).T
        
        # Additional filtering
            if ax is not None and az is not None:
                ga = self._image_gaussian(ga, ax, az, 'LOW_PASS')  # To be implemented
                ga=ga[0]
                ga = self._normali(ga)
            
        # Calculate error
            Ea = np.sum(xab_**2) / xab_.size
        
        # Step size calculation (placeholder)
            step_a = self._cross_step_a(a, b, ga, Ea, ka, no_print)  # To be implemented
            da = -step_a * ga
            a += da
            da_ += da
        
        # ------------------------- b update -------------------------
        # Forward calculation
            xab_ = self._cross2d(a, b, Dx, Dz)
        
        # Gradient calculation
            Jb = self._crossJb(a, Dx, Dz)
            # npp=Jb.toarray()
            gb = Jb @ xab_.ravel(order='F')
        
        # Hessian and filtering
            if ax is not None and az is not None:
                gb = gb.reshape(b.T.shape).T
                gb = self._image_gaussian(gb, ax, az, 'LOW_PASS')  # To be implemented
                gb=gb[0]
                gb = gb.ravel(order='F')
            
            gb = self._normali(gb)
        
        # Solve linear system
            B = Jb @ Jb.T + step_b * eye(b.size, format='csr')
            gb = spsolve(B, gb).reshape(b.T.shape).T
        
        # Additional filtering
            if ax is not None and az is not None:
                gb = self._image_gaussian(gb, ax, az, 'LOW_PASS')  # To be implemented
                gb=gb[0]
                gb = self._normali(gb)
            
        # Calculate error
            Eb = np.sum(xab_**2) / xab_.size
        
        # Step size calculation (placeholder)
            step_b = self._cross_step_b(a, b, gb, Eb, kb, no_print)  # To be implemented
            db = -step_b * gb
            b += db
            db_ += db
        
        # Update iteration
            i_ += 1
            E_ = 0.5 * (Ea + Eb)
        
        # Print progress
            if i_ % max(1, int(ni * 0.1)) == 1:
                print(f'   xgrad error {E_:.2e} at iteration {i_}')
    
        return a, b, da_, db_
    
        
    def _cross_step_a(self, a, b, ga, Ea, ka, no_print=False):
        # 计算不同步长的误差
        Ea_ = np.zeros(len(ka))
        for i_ in range(len(ka)):
            a_tmp = a - ka[i_] * ga
            xab_tmp = self._cross2d(a_tmp, b)
            Ea_[i_] = np.sum(xab_tmp**2) / xab_tmp.size
        
        # 合并基准误差
        Ea_full = np.concatenate([[Ea], Ea_])
        ka_full = np.concatenate([[0], ka])
        
        # 二次曲线拟合
        try:
            p = np.polyfit(ka_full, Ea_full, 2)
            step_a = -p[1]/(2*p[0])  # 极值点公式
        except:
            step_a = 0
        
        # 异常处理
        if not no_print:
            if np.isnan(step_a) or step_a == 0:
                # print(f'    step a is zero or nan = {step_a:.2f}')
                # print('    your inversion is pretty screwed, lol')
                # print('    try changing percentages of step-sizes. parabola is flat :(')
                step_a = 0
            elif step_a < 0:
                # print(f'    step a is negative = {step_a:.2f}')
                # print('    try changing percentages of step-sizes and look at parabola')
                step_a = 0
            else:
                # print(f'    step a {step_a:.2f}')
                pass
        return step_a


    def _cross_step_b(self, a, b, gb, Eb, kb, no_print=False):
        # 计算不同步长的误差
        Eb_ = np.zeros(len(kb))
        for i_ in range(len(kb)):
            b_tmp = b - kb[i_] * gb
            xab_tmp = self._cross2d(a, b_tmp)
            Eb_[i_] = np.sum(xab_tmp**2) / xab_tmp.size
        
        # 合并基准误差
        Eb_full = np.concatenate([[Eb], Eb_])
        kb_full = np.concatenate([[0], kb])
        
        # 二次曲线拟合
        try:
            p = np.polyfit(kb_full, Eb_full, 2)
            step_b = -p[1]/(2*p[0])  # 极值点公式
        except:
            step_b = 0
        
        # 异常处理
        if not no_print:
            if np.isnan(step_b) or step_b == 0:
                # print(f'    step b is zero or nan = {step_b:.2f}')
                # print('    your inversion is pretty screwed, lol')
                # print('    try changing percentages of step-sizes. parabola is flat :(')
                step_b = 0
            elif step_b < 0:
                # print(f'    step b is negative = {step_b:.2f}')
                # print('    try changing percentages of step-sizes and look at parabola')
                step_b = 0
            else:
                # print(f'    step b {step_b:.2f}')
                pass
        
        return step_b
    
    
    def _cross_a(self, a, b, ni, tol_, ka, Dx, Dz, ax=None, az=None):
        no_print = False
        step_a = 1.0
        da_ = np.zeros_like(a)
        Ja = self._crossJa(b, Dx, Dz)
        E_ = np.inf
        i_ = 0
    
        while E_ > tol_ and i_ < ni:
        # ------------------------- a update -------------------------
        # Forward calculation
            xab_ = self._cross2d(a, b, Dx, Dz)
        
        # Gradient calculation

            # npp=Ja.toarray()
            ga = Ja @ xab_.ravel(order='F')
        
        # Hessian and filtering (placeholder)
            if ax is not None and az is not None:
                ga = ga.reshape(a.T.shape).T
                ga = self._image_gaussian(ga, ax, az, 'LOW_PASS')  # To be implemented
                ga=ga[0]
                ga = ga.ravel(order='F')
            ga = self._normali(ga)
        
        # Solve linear system
            A = Ja @ Ja.T + step_a * eye(a.size, format='csr')
            ga = spsolve(A, ga).reshape(a.T.shape).T
        
        # Additional filtering
            if ax is not None and az is not None:
                ga = self._image_gaussian(ga, ax, az, 'LOW_PASS')  # To be implemented
                ga=ga[0]
                ga = self._normali(ga)
            
        # Calculate error
            Ea = np.sum(xab_**2) / xab_.size
        
        # Step size calculation (placeholder)
            step_a = self._cross_step_a(a, b, ga, Ea, ka, no_print)  # To be implemented
            da = -step_a * ga
            a += da
            da_ += da

       
        # Update iteration
            i_ += 1
            E_ = (Ea)
        
        # Print progress
            if i_ % max(1, int(ni * 0.1)) == 1:
                print(f'   xgrad error {E_:.2e} at iteration {i_}')    
    
        return a, da_   
    
    
    
    
    def _cross_b(self, a, b, ni, tol_, kb, Dx, Dz, ax=None, az=None):
        no_print = False
        step_b = 1.0
        db_ = np.zeros_like(b)
        Jb = self._crossJb(a, Dx, Dz)
        E_ = np.inf
        i_ = 0
    
        while E_ > tol_ and i_ < ni:       
        # ------------------------- b update -------------------------
        # Forward calculation
            xab_ = self._cross2d(a, b, Dx, Dz)
        
            # npp=Jb.toarray()
            gb = Jb @ xab_.ravel(order='F')
        
        # Hessian and filtering
            if ax is not None and az is not None:
                gb = gb.reshape(b.T.shape).T
                gb = self._image_gaussian(gb, ax, az, 'LOW_PASS')  # To be implemented
                gb=gb[0]
                gb = gb.ravel(order='F')
            
            gb = self._normali(gb)
        
        # Solve linear system
            B = Jb @ Jb.T + step_b * eye(b.size, format='csr')
            gb = spsolve(B, gb).reshape(b.T.shape).T
        
        # Additional filtering
            if ax is not None and az is not None:
                gb = self._image_gaussian(gb, ax, az, 'LOW_PASS')  # To be implemented
                gb=gb[0]
                gb = self._normali(gb)
            
        # Calculate error
            Eb = np.sum(xab_**2) / xab_.size
        
        # Step size calculation (placeholder)
            step_b = self._cross_step_b(a, b, gb, Eb, kb, no_print)  # To be implemented
            db = -step_b * gb
            b += db
            db_ += db
        
        # Update iteration
            i_ += 1
            E_ = (Eb)
        
        # Print progress
            if i_ % max(1, int(ni * 0.1)) == 1:
                print(f'   xgrad error {E_:.2e} at iteration {i_}')
    
        return b, db_    
    
    
    
    
    
    
    
    