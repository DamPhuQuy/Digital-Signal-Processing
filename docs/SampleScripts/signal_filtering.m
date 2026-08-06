% Tao tin hieu la tong cac song sin,
% sau do loc tin hieu dung bo loc thong thap
% cuoi cung, phat t/h truoc va sau khi loc ra loa

% cac tham so co ban (co the thay doi va quan sat/nghe ket qua)
fs = 10000; % tan so lay mau (Hz)
order = 100; % bac cua bo loc FIR (bang chieu dai cua h[n])
fc = 2000;  % tan so cat cua bo loc thong thap (Hz)
dur = 2; % do dai tin hieu (giay)
t = 0:1/fs:dur; % truc thoi gian bi roi rac hoa

% cac gioi han truc x khi ve do thi
Fx_max = 10000; % tan so cuc dai tren do thi pho (Hz)
Tx_max = 0.02;  % thoi gian cuc dai tren do thi tin hieu (giay)

figure(1);
% tao tin hieu vao la tong 3 song sin: 500 Hz, 2000, va 3500 Hz
x = 3*sin(2*pi*500*t) + 2*sin(2*pi*2000*t) + sin(2*pi*3500*t);
subplot(2,1,1);
plot(t,x); 
title('Tin hieu vao x[n]');
xlabel('Thoi gian t (giay)');
ylabel('Bien do');
axis([0, Tx_max, min(x), max(x)]);

% thiet ke bo loc FIR thong thap dung ham fir1() cua MATLAB
% SV tu sua CT de su dung lai bo loc da thiet ke trong BT truoc
h =  fir1(order,fc/fs,'low'); % h[n] cua bo loc thong thap voi t/s cat fc

% loc tin hieu vao thu duoc tin hieu ra
y = filter(h,1,x); % ham loc
subplot(2,1,2);
plot(t,y); 
title('Tin hieu ra y[n]');
xlabel('Thoi gian t (giay)');
ylabel('Bien do');
axis([0, Tx_max, min(y), max(y)]);

% ve pho t/h vao, DUBD cua bo loc, va pho t/h ra
figure(2);
N = 1024; % so diem tren truc tan so w
w = 0:pi/N:pi;
% chuyen tu t/s ly thuyet w (radian) sang t/s thuc F (Hz)
F = fs * w / (2*pi);

X = freqz(x,1,w); % nhan duoc X(ejw)
modX = abs(X)/fs; % chia cho he so ti le fs
subplot(3,1,1);
plot(F,modX);grid;
axis([0, fs/2 , 0, max(modX)]);
title ('Pho bien do tin hieu vao');
xlabel('F(Hz)');

H=freqz(h,1,w); % nhan duoc H(ejw)
subplot(3,1,2);
plot(F,abs(H),'red');grid;
axis([0, fs/2 , 0, max(abs(H))]);
title('Dap ung bien do cua bo loc H(F)');
xlabel('Tan so F (Hz)');
ylabel('Bien do');

Y = freqz(y,1,w); % nhan duoc Y(ejw)
modY = abs(Y)/fs; % chia cho he so ti le fs
subplot(3,1,3);
plot(F,modY);grid;
axis([0, fs/2 , 0, max(modY)]);
title ('Pho bien do tin hieu ra');
xlabel('F(Hz)');

% phat tin hieu vao/ra ra loa (thuc hien khoi phuc tin hieu)
sound(x, fs, 16);
pause(4)
sound(y, fs, 16);