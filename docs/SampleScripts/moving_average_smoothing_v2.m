% chuong trinh lam tron 1 tin hieu de khu nhieu (denoise)
% dung bo loc lay trung binh cong 3 diem (3-points moving-averaging filter).
clf;                            % clear figures
L = 16000;                      % do dai tin hieu (in samples)
n = 0:L-1;                      % bien thoi gian roi rac
d = 0.1*randn(1,L);             % sinh tin hieu Gausian noise d[n]
s = 2*n.*(0.9.^n);              % sinh tin hieu goc s[n] = 2n(0.9)^n
x = s + d;                      % tin hieu co nhieu x[n]=s[n]+d[n]

figure(1)                       % tao figure 1
hold on
subplot(2,1,1)                  % tao luoi 3x1 plots va ve plot o vi tri #1
plot(n,d,'r-',n,s,'k--',n,x,'b-.'); % ve do thi d[n],s[n],x[n]
xlabel('Chi so thoi gian n');
ylabel('Bien do');
legend('d[n]','s[n]','x[n]');
title('Noise vs. original vs. noisy signals');

% dung ham cai dat bo loc filter() de loc nhieu
a = [1];
b = [1/3, 1/3, 1/3];
y = filter(b, a, x);

subplot(2,1,2)                  % tao luoi 3x1 plots va ve plot o vi tri #1
plot(n,x,'k--',n,y,'b-.'); % ve do thi x[n] vs. y[n]
xlabel('Chi so thoi gian n');
ylabel('Bien do');
legend('x[n]','y[n]');
title('Noisy vs filtered signals');

figure(2)
% plot freq response, given its expression
w = -pi:pi/100:pi;          % vecto tan so (truc t/s)
H = 1/3*(1 + exp(-j*w) + exp(-j*2*w));   % dap ung tan so
Habs = abs(H);              % dap ung bien do
Hphase = angle(H);          % dap ung pha

% gia su biet ts lay mau
Fs=16000;  % in Hz
f = w * Fs / 2 / pi;
subplot(2,1,1);
plot(f,Habs);
grid;
title('Dap ung bien do');
xlabel('Frequency (Hz)');

subplot(2,1,2);
plot(f,Hphase);
grid;
title('Dap ung pha');
xlabel('Frequency (Hz)');


sound(x,Fs);
pause(4);
sound(y,Fs);






