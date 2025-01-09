import numpy as np
import matplotlib.pyplot as plt
import GooseEYE
# import porespy as ps
origin = np.fromfile("label.raw",dtype=np.uint8).reshape((-1,256,256))
origin = np.where(origin==0,np.ones_like(origin),np.zeros_like(origin))
segnet = np.fromfile("segnet.raw",dtype=np.uint8).reshape((-1,256,256))
segnet = np.where(segnet==0,np.ones_like(segnet),np.zeros_like(segnet))
rsegnet = np.fromfile("rsegnet.raw",dtype=np.uint8).reshape((-1,256,256))
rsegnet = np.where(rsegnet==0,np.ones_like(rsegnet),np.zeros_like(rsegnet))
unet = np.fromfile("unet.raw",dtype=np.uint8).reshape((-1,256,256))
unet = np.where(unet==0,np.ones_like(unet),np.zeros_like(unet))
runet = np.fromfile("runet.raw",dtype=np.uint8).reshape((-1,256,256))
runet = np.where(runet==0,np.ones_like(runet),np.zeros_like(runet))
scpgan = np.fromfile("scpseg.raw",dtype=np.uint8).reshape((-1,256,256))
scpgan = np.where(scpgan==0,np.ones_like(scpgan),np.zeros_like(scpgan))
    # I = origin1[i]
    # I1 = result1[i]

    # C = GooseEYE.clusters(I)
    # C1 = GooseEYE.clusters(I1)
    # res =GooseEYE.C2((1, 513), C,C)
    # res1 =GooseEYE.C2((1, 513), C1,C1)
    # Co1 = GooseEYE.clusters(origin1[i])
    # Cr1 = GooseEYE.clusters(result1[i])
    # reso1 =GooseEYE.C2((1, 513), Co1, Co1)
    # res1 =GooseEYE.C2((1, 513), Cr1, Cr1)
    # ano1[i] = reso1[0,256:376]
    # anr1[i] = res1[0,256:376]
    # Co2 = GooseEYE.clusters(origin2[i])
    # Cr2 = GooseEYE.clusters(result2[i])
    # reso2 =GooseEYE.C2((1, 513),Co2, Co2)
    # res2 =GooseEYE.C2((1, 513), Cr2, Cr2)
    # ano2[i] = reso2[0,256:376]
    # anr2[i] = res2[0,256:376]
    # Co3 = GooseEYE.clusters(origin3[i])
    # Cr3 = GooseEYE.clusters(result3[i])
    # reso3 =GooseEYE.C2((1, 513), Co3, Co3)
    # res3 =GooseEYE.C2((1, 513), Cr3, Cr3)
    # ano3[i] = reso3[0,256:376]
    # anr3[i] = res3[0,256:376]
# Co1 = GooseEYE.clusters(origin)
# reso1 =GooseEYE.C2((1, 513),Co1, Co1)
# Co2 = GooseEYE.clusters(srcnn)
# reso2 =GooseEYE.C2((1, 513), Co2, Co2)
# Co3 = GooseEYE.clusters(srgan)
# reso3 =GooseEYE.C2((1, 513), Co3, Co3)
# Co4 = GooseEYE.clusters(scpgan)
# reso4 =GooseEYE.C2((1, 513), Co4, Co4)
ano1 = np.zeros((200,120))
ano2 = np.zeros((200,120))
ano3 = np.zeros((200,120))
ano4 = np.zeros((200,120))
ano5 = np.zeros((200,120))
ano6 = np.zeros((200,120))
for i in range(200):

    # reso1 =GooseEYE.L((1, 513),origin[i])
    # reso2 =GooseEYE.L((1, 513), segnet[i])
    # reso3 =GooseEYE.L((1, 513), rsegnet[i])
    # reso4 =GooseEYE.L((1, 513), unet[i])
    # reso5 =GooseEYE.L((1, 513),runet[i])
    # reso6 =GooseEYE.L((1, 513), scpgan[i])
    Co1 = GooseEYE.clusters(origin[:,i])
    reso1 =GooseEYE.C2((1, 513),Co1, Co1)
    # reso1 =GooseEYE.S2((1, 513),origin[i], origin[i])
    Co2 = GooseEYE.clusters(segnet[:,i])
    reso2 =GooseEYE.C2((1, 513), Co2, Co2)
    # reso2 =GooseEYE.S2((1, 513), segnet[i], segnet[i])
    Co3 = GooseEYE.clusters(rsegnet[:,i])
    reso3 =GooseEYE.C2((1, 513), Co3, Co3)
    # reso3 =GooseEYE.S2((1, 513), rsegnet[i], rsegnet[i])
    Co4 = GooseEYE.clusters(unet[:,i])
    reso4 =GooseEYE.C2((1, 513), Co4, Co4)
    # reso4 =GooseEYE.S2((1, 513), unet[i], unet[i])
    Co5 = GooseEYE.clusters(runet[:,i])
    reso5 =GooseEYE.C2((1, 513), Co5, Co5)
    # reso5 =GooseEYE.S2((1, 513), runet[i], runet[i])
    Co6 = GooseEYE.clusters(scpgan[:,i])
    reso6 =GooseEYE.C2((1, 513), Co6, Co6)
    # reso6 =GooseEYE.S2((1, 513), scpgan[i], scpgan[i])
    ano1[i] = reso1[0, 256:376]
    ano2[i] = reso2[0, 256:376]
    ano3[i] = reso3[0, 256:376]
    ano4[i] = reso4[0, 256:376]
    ano5[i] = reso5[0, 256:376]
    ano6[i] = reso6[0, 256:376]
    print(i)

reo1 = np.mean(ano1, axis=0)
reo2 = np.mean(ano2, axis=0)
reo3 = np.mean(ano3, axis=0)
reo4 = np.mean(ano4, axis=0)
reo5 = np.mean(ano5, axis=0)
reo6 = np.mean(ano6, axis=0)



from matplotlib import rcParams
config = {
    "font.family":'serif',
    "font.size": 10.5,
    "mathtext.fontset":'stix',
    "font.serif": ['SimSun'],
}
rcParams.update(config)
plt.plot(np.arange(120),reo1,label = '高分辨率分割图像')
plt.plot(np.arange(120),reo2,label = 'SegNet网络分割结果',linestyle='--')
plt.plot(np.arange(120),reo3,label = 'ResSegNet网络分割结果',linestyle='--')
plt.plot(np.arange(120),reo4,label = 'UNet网络分割结果',linestyle='--')
plt.plot(np.arange(120),reo5,label = 'ResUNet网络分割结果',linestyle='--')
plt.plot(np.arange(120),reo6,label = 'SPCGAN网络分割结果',linestyle='--')

plt.xlabel("距离 r", size=10.5)
plt.ylabel("$C_2$(r)",fontproperties='Times New Roman', size=10.5)
plt.legend()
plt.savefig("C2_x.png", bbox_inches="tight", pad_inches=0.1,dpi=300)
plt.show()