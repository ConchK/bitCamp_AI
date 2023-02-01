



#제출
# y_submit=model.predict(test)
# print(y_submit)
# print(y_submit.shape)  #(715, 1)

#.to_csv()를 사용해서
#submission_0105.csv를 완성
# print(submission)
# submission['count']= y_submit
# print(submission)
#
# submission.to_csv(path+ 'submission_01112300.csv')  #제출용 파일 생성

# print("=====================================")
# print(hist) 
# print(hist.history)
# print("=====================================")

# import matplotlib.pyplot as plt

# plt.figure(figsize=(9,6))
# plt.plot(hist.history['loss'], c='red', marker='.', label='loss')   #리스트 형태는 x를 명시 하지 않아도 됨. 어차피 앞에서 부터.
# plt.plot(hist.history['val_loss'], c='blue', marker='.', label='val_loss')  
# plt.grid()
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.title('boston loss')
# plt.legend(loc='upper left')  #location지정하지 않으면 그래프가 없는 지점에 자동으로 생성

# plt.show()