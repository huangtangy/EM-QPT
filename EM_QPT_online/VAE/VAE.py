#==========================================================#
#========log:18 Feb 2025 for 1-q gate operation ===========#
#==========================================================#
from model import *
import torch
from qutip import basis, qeye, sigmax, sigmay, sigmaz, tensor, ket2dm, Qobj, fidelity, rand_ket
from itertools import product
from brokenaxes import *

def is_CPTP_chi_np(chi_matrix, N, tol=1e-5):
    """
    判断 NumPy Chi 矩阵是否是 CPTP
    """
    dim2 = chi_matrix.shape[0]
    dim = int(dim2 ** 0.5)

    pauli = [qeye(2), sigmax(), sigmay(), sigmaz()]
    pauli_basis = [tensor(*op) for op in product(pauli, repeat=N)]
    pauli_basis = [p.full() for p in pauli_basis]

    # dim = 2
    # pauli_basis = np.stack([
    #     np.eye(dim, dtype=np.complex64),  # I
    #     np.array([[0, 1], [1, 0]], dtype=np.complex64),  # X
    #     np.array([[0, -1j], [1j, 0]], dtype=np.complex64),  # Y
    #     np.array([[1, 0], [0, -1]], dtype=np.complex64)  # Z
    # ])

    # 1. 赫米特性检查
    # print(np.allclose(chi_matrix, chi_matrix.T.conj(), atol=tol))
    if not np.allclose(chi_matrix, chi_matrix.T.conj(), atol=tol):
        return False

    # 2. 正定性检查
    eigenvalues = np.linalg.eigvalsh(chi_matrix)
    # print('eigenvalues',eigenvalues)
    if np.any(eigenvalues < -tol):
        return False

    # 3. 保持迹检查
    identity = np.eye(dim, dtype=np.complex64)
    trace_test = sum(chi_matrix[m, n] * pauli_basis[n].conj().T @ pauli_basis[m]
                     for m in range(dim2) for n in range(dim2))

    if not np.allclose(trace_test, identity, atol=tol * 100):
        # print('trace perseving',np.allclose(trace_test, identity, atol=tol),trace_test)
        return False

    return True

def clean_cholesky(img):
    """
    清洗输入矩阵，得到用于 Cholesky 分解的矩阵 T

    Args:
        img (torch.Tensor): 形状为 (batch_size, hilbert_size, hilbert_size, 2)
                            的张量，表示神经网络的随机输出。
                            最后一个维度用于分离实部和虚部。

    Returns:
        T (torch.Tensor): 形状为 (N, hilbert_size, hilbert_size) 的张量，
                          表示 N 个用于 Cholesky 分解的矩阵（复数张量）。
    """
    # 分离实部和虚部
    real = img[:,0,:, :]
    imag = img[:,1,:, :]

    # 取虚部的对角线元素
    diag_all = torch.diagonal(imag, dim1=1, dim2=2)  # shape: (batch_size, hilbert_size)
    # 构造对角矩阵
    diags = torch.diag_embed(diag_all)

    # 将虚部对角线元素置零
    imag = imag - diags

    # 取下三角部分
    real = torch.tril(real)
    imag = torch.tril(imag)

    # 构造复数矩阵
    T = torch.complex(real, imag)
    return T

def density_matrix_from_T(tmatrix):
    """
    从 T 矩阵得到密度矩阵，并进行归一化

    Args:
        tmatrix (torch.Tensor): 形状为 (N, hilbert_size, hilbert_size)
                                 的张量，表示 N 个有效的 T 矩阵（复数张量）。

    Returns:
        rho (torch.Tensor): 形状为 (N, hilbert_size, hilbert_size) 的张量，
                            表示 N 个密度矩阵。
    """
    T = tmatrix
    # 计算共轭转置 T†
    T_dagger = T.transpose(-1, -2).conj()

    # 计算 T_dagger @ T
    proper_dm = torch.matmul(T_dagger, T)

    # 计算每个矩阵的迹
    traces = torch.einsum('bii->b', proper_dm)
    # 计算归一化因子 1/trace，并调整形状便于广播
    inv_traces = 1.0 / traces
    inv_traces = inv_traces.view(-1, 1, 1)

    # 归一化密度矩阵
    rho = proper_dm * inv_traces

    # rho = rho.view(rho.shape[0], rho.shape[1]*rho.shape[2])
    return rho

# 计算 VAE 的损失
# 计算重构损失 + KL 散度

def loss_function(recon_x, x, mu, logvar):
    BCE = F.mse_loss(recon_x, x, reduction='sum')
    KL_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE +  KL_div

# 训练 VAE
def train_vae(model, train_loader, optimizer, epochs=10, device='cuda'):
    model.train()
    losslist=[]
    for epoch in range(epochs):
        train_loss = 0
        for batch_idx, (data,) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        losslist.append(train_loss)
        print(f"Epoch {epoch + 1}, Loss: {train_loss / len(train_loader.dataset):.4f}")
    return  losslist
# 测试 VAE

def test_vae(model, test_loader, device='cuda'):
    z_space = []
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, in test_loader:
            #data = data.to(device)
            recon, mu, logvar = model(data)# mu is the latant vector
            #z = model.reparameterize(mu, logvar)
            z_space.append(mu.numpy())
            loss = loss_function(recon, data, mu, logvar)
            test_loss += loss.item()
    print(f"Test Loss: {test_loss / len(test_loader.dataset):.4f}")
    return z_space
# 运行 VAE 训练和测试


if __name__ == "__main__":
    # 运行 VAE 训练和测试
    trainnum = 2**10
    batch_size = 16
    epochs = 100
    latent_dim = 2
    p = 0.05
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建模拟数据 (128, 2, 4, 4) -> 展平为 (128, 16)

    num_data = np.load('./trainingdata/fig_id_005p_1q_2048.npz')['fig_id_gan']#'fig_id_gan_01p_1q_1024.npz'
    num_data1 = num_data[:, :, :, :]
    #num_data2 = num_data1.reshape(len(num_data1),2, 16)

    dim = 10
    num_data = np.zeros(shape=(len(num_data1), 2, dim))
    for ii in range(len(num_data)):
        data0 = num_data1[ii]
        newdata = np.zeros(dim)
        newdata1 = np.zeros(dim)
        jj = 0
        # 提取一阶梯的错误（主要部分）
        for i in range(4):
            for j in range(4-i):
                ele = data0[0, i, j+i]
                ele1 = data0[1, i, j+i]
                newdata[jj] = ele
                newdata1[jj] = ele1
                jj += 1
        num_data[ii, 0, :] = newdata
        num_data[ii, 1, :] = newdata1
    input = 16
    #num_data = num_data.reshape(1024,2,10)#num_data2[:, :, :input]
    num_data = num_data1.reshape(len(num_data1),2,16)
    data = num_data1[:trainnum, :, :]

    train_data = torch.from_numpy(data).float()#.view(-1, 7)
    # train_data = TensorDataset(data)

    # train_data = torch.rand(1000, 2, 4, 4).view(-1, 16)
    # test_data = torch.rand(200, 2, 4, 4).view(-1, 16)
    testnum = 1000
    data_test =  data[:testnum, :, :]
    test_data = torch.from_numpy(data_test).float()
    # test_data = TensorDataset(data)
    train_loader = DataLoader(TensorDataset(train_data), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_data), batch_size=testnum, shuffle=True)

    model = VAE(latent_dim,input).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    losslist = train_vae(model, train_loader, optimizer, epochs, device)

    z_space  = test_vae(model, test_loader, device)

    plt.figure()
    plt.plot(losslist)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("VAE Training Loss")
    plt.show()

    plt.figure()
    # #x = np.arange(len(z_space[0]))
    if len(z_space[0][0]) > 0:
        z  = TSNE(n_components=2).fit_transform(z_space[0])
    else:
        z =z_space[0]
    x,y = [z[0] for z in z_space[0]], [z[1] for z in z_space[0]]# t-SNE 降维潜在空间
    # plt.scatter(x,y)
    # plt.xlabel("latant_dim0")
    # plt.ylabel("latant_dim0")
    # plt.legend()
    # plt.title("latant space")
    # plt.show()
    # 计算密度估计
    x,y = np.array(x),np.array(y)
    xy = np.vstack([x, y])
    density = gaussian_kde(xy)(xy)
    density /=np.max(density)

    # 创建密度等高线图
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.set_title("Latent Space (N="+str(trainnum)+")")
    ax.set_xlabel("Latent Dimension 1")
    ax.set_ylabel("Latent Dimension 2")
    ax.axhline(0, color='gray', linestyle='--', alpha=0.7)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.7)
    # 绘制散点并用颜色表示密度
    sc = ax.scatter(x, y, c=density, cmap="Blues", s=5, alpha=0.5)
    # 生成等高线
    xmin, xmax =  x.min() *2, x.max() *2
    ymin, ymax =  y.min() *2, y.max()*2
    X, Y = np.meshgrid(np.linspace(xmin, xmax, 1000), np.linspace(ymin, ymax, 1000))
    positions = np.vstack([X.ravel(), Y.ravel()])
    Z = gaussian_kde(xy)(positions).reshape(X.shape)
    Z /= np.max(Z)
    ax.contour(X, Y, Z, levels=6, cmap="Blues")
    # 添加颜色条
    plt.xticks([0])
    plt.yticks([0])
    plt.colorbar(sc, label="Density")
    plt.show()

    #=========model evaluation=======
    torch.save(model.state_dict(), './saved_model/1q/model_weights'+str(trainnum)+'p='+str(p)+'.pth')
    model = VAE(latent_dim,input)
    model.load_state_dict(torch.load('./saved_model/1q/model_weights'+str(trainnum)+'p='+str(p)+'.pth'))
    model.eval()
    with torch.no_grad():
        z = torch.randn(400, latent_dim)#.to(device)  # 随机采样 latent vector
        sample0 = model.decode(z)
        sample1 = clean_cholesky(sample0.reshape(400,2,4,4))
        sample2 = density_matrix_from_T(sample1)
        sample2_numpy = sample2.cpu().numpy()
        print('is it Hermitian?',np.allclose(sample2_numpy[0],sample2_numpy[0].T.conj(),atol=1e-5))
        print('is it CPTP?',is_CPTP_chi_np(sample2_numpy[0],1 ))
        # np.allclose(sample2_numpy[0],sample2_numpy[0].T.conj(),atol=1e-5) # 判断Hermitial
        sample = sample0.cpu().numpy()
    print("Generated Sample Shape:", np.shape(sample))

    #np.savez('digital_twin'+str(trainnum)+'.npz',sample=sample )

    images = num_data[:100, :, :]

    # def reconstruct_images(im):
    #     new_im = np.zeros(shape=(2,4,4))
    #     jj = 0
    #     # 提取一阶梯的错误（主要部分）
    #     for i in range(4):
    #         for j in range(4 - i):
    #             new_im[0, i, j + i] = im[0,jj]
    #             new_im[1, i, j + i] = im[1,jj]
    #             if i!=j:
    #                 new_im[0,j + i,i] = -im[0, jj]
    #                 new_im[1, j + i,i] = -im[1, jj]
    #             jj += 1
    #     return new_im

    # vmi, vmx = -0.01, 0.01
    # plt.figure()
    # for i in range(16):
    #     plt.subplot(4, 4, i + 1)
    #     plt.plot(images[i][0])  #
    #     plt.plot(sample[i][0],color='r')  #
    #     plt.ylim([0,1])
    #     plt.axis('off')
    #     # plt.colorbar()
    # plt.show()
    plt.figure()
    fig, axes = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(10):
        recons_data = sample[i][0].reshape(4,4)#sample[i]
        #recons_data = reconstruct_images(recons_data)
        axes[0, i].imshow(num_data1[i, 0], cmap="gray")  # 原始输入
        axes[1, i].imshow(recons_data, cmap="gray")  # 生成输出
        axes[0, i].axis("off")
        axes[1, i].axis("off")
    plt.suptitle("Top: Original | Bottom: Reconstructed (second layer)")
    plt.show()

    # plt.figure()
    # for i in range(16):
    #     plt.subplot(4, 4, i + 1)
    #     plt.plot(images[i][1])  #
    #     plt.plot(sample[i][1],color='r')  #
    #     plt.ylim([-0.01,0.01])
    #     # plt.axis('off')
    #     # plt.colorbar()
    # plt.show()
    plt.figure()
    fig, axes = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(10):
        #recons_data = sample[i]
        recons_data = sample[i][1].reshape(4,4)
        #recons_data = reconstruct_images(recons_data)
        axes[0, i].imshow(num_data1[i, 1], cmap="gray")  # 原始输入
        axes[1, i].imshow(recons_data, cmap="gray")  # 生成输出
        axes[0, i].axis("off")
        axes[1, i].axis("off")
    plt.suptitle("Top: Original | Bottom: Reconstructed (second layer)")
    plt.show()