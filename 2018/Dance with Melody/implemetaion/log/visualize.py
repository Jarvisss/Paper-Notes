from tensorboard.backend.event_processing import event_accumulator
from matplotlib import pyplot
# 加载日志数据
ea = event_accumulator.EventAccumulator('./LSTM-AE_rotate_Ortho_Leaky_Temporal_InputSize_50_Seq_120_TempoNor_Threshold_0.450_Masking_Reduced_10')
ea.Reload()
print(ea.scalars.Keys())

val_psnr = ea.scalars.Items('Valid/Loss')
train_psnr = ea.scalars.Items('Train/Loss')
print(len(val_psnr))
print([(i.step, i.value) for i in val_psnr])
# fig = pyplot.figure()
pyplot.plot([i.step for i in train_psnr],[i.value for i in train_psnr],'r')
pyplot.plot([i.step for i in val_psnr],[i.value for i in val_psnr],'b')
pyplot.show()
# fig.show()