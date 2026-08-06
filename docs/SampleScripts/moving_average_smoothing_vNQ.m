% chuong trinh lam tron 1 tin hieu de khu nhieu (denoise)
% dung bo loc trung binh truot 3 diem (3-points moving-averaging filter)
% co PTSP: y[n] = 1/3*(x[n]+x[n-1]+x[n-2]) (he nhan qua) 
%-> h[n]=1/3*(delta[n]+delta[n-1]+delta[n-2]) = [1/3, 1/3, 1/3] (n=0,1,2)
%
% problem statement: co tin hieu bi lan voi nhieu x[n], di khoi phuc lai
% tin hieu goc s[n] (an trong x[n])
% ky vong KQ: y[n] cang giong s[n] cang tot

% sinh tin hieu bi lan voi nhieu cong (additive noise)
clear
close all                        % close all figures
A = 0.5;                        % cong suat nhieu ti le voi sqr(A)  
L = 100;                         % do dai tin hieu
n = 0:L-1;                      % bien thoi gian roi rac
d = A*randn(1,L);             % sinh tin hieu Gaussian white noise d[n] (A=std cua tin hieu nhieu)
s = 2*n.*(0.9.^n);              % sinh tin hieu goc s[n] = 2n(0.9)^n
x = s + d;                      % tin hieu co nhieu x[n]=s[n]+d[n]

figure(1)                    
subplot(3,1,1)                 
plot(n,d,'r-',n,s,'k--',n,x,'b-.'); % ve do thi d[n],s[n],x[n]
xlabel('Chi so thoi gian n');
ylabel('Bien do');
legend('d[n]','s[n]','x[n]');
title('Noise d[n] vs. original s[n] vs. noisy signals x[n]');

% cach 1: cai dat he thong bang cach dich thoi gian va tinh
% TBC cua 3 tin hieu --> cach nay chi dung dc khi co duoc toan bo x[n],
% chi cai dat offline (nhan toan bo gia tri cua x[n])
x1 = [x];                 % x1[n] = x[n]
x2 = [0, x(1:L-1)];   % x2[n] = x[n-1]
x3 = [0, 0, x(1:L-2)]; % x3[n] = x[n-2]

subplot(3,1,2)
plot(n,x1,'r-.',n,x2,'b-.',n,x3,'k-.');
xlabel('Chi so thoi gian n');
ylabel('Bien do');
legend('x[n]','x[n-1]','x[n-2]');
title('time-shifted signals of x[n]');

y1 = 1/3*(x1+x2+x3);     % cai dat PTSP
%subplot(3,1,3)
% ve do thi y1[n] vs. s[n]
% plot(n,y1(1:L),'r-',n,s(1:L),'b-');
% xlabel('Chi so thoi gian n');
% ylabel('Bien do');
% legend('y1[n]','s[n]');
% title('3-points smoothed y1[n] vs. original signal s[n]');

% cach 2: cai dat he thong bang cach dung ham tinh tong chap conv()
hNQ = 1/3 * ones(1,3);    % hNQ[n] = [1/3, 1/3, 1/3] (n=0,1,2)
y2 = conv(x, hNQ);          % y2[n] = x[n] * hNQ[n]  --> L(y2) = L(x) + L(hNQ) - 1 = L(x) + 2
% ve do thi y2[n] vs. s[n]
% figure(2)
% plot(n,y2(1:L),'r-',n,s(1:L),'b-');
% xlabel('Chi so thoi gian n');
% ylabel('Bien do');
% legend('y2[n]','s[n]');
% title('3-points smoothed y2[n] vs. original signal s[n]');

% cach 3: dung vong lap (hieu n la mot chi so mau nao do) --> truc quan
% nhat, co the cai dat theo real-time (online mode)
y3 = zeros(1,L);
y3(1) = (1/3)*(x(1));  % x[0] = x[-1] = 0 (zero padding)
y3(2) = (1/3)*(x(2)+x(1));
for i=3:L
    y3(i) = (1/3)*(x(i)+x(i-1)+x(i-2)); % PTSP
end

% cach 4: dung ham filter() --> tien dung va pho bien nhat vi ap dung dc
% voi moi PTSP cua he NQ, cai dat thuat toan tinh nhanh tren mien tan so
% FFT va IFFT
b = 1/3*ones(1,3);
a = [1];
y4 = filter(b,a,x);  %--> L(y4) = L(x)

% cai dat bo loc voi kich thuoc N=5 co PTSP: y[n] = 1/5*(x[n]+x[n-1]+x[n-2]+x[n-3]+x[n-4]) 
b = 1/5*ones(1,5);
a = [1];
y5 = filter(b,a,x);  %--> L(y5) = L(x)

% ve do thi xep chong x[n], s[n] va y4[n] va y5[n] de test ket qua
figure(3)
hold all
plot(n(1:L-2),x(1:L-2),'b-');
plot(n(1:L-2),s(1:L-2),'g-');
plot(n(1:L-2),y4(2:L-1),'k+-');
plot(n(1:L-2),y5(3:L),'r.-');
hold off
xlabel('Chi so thoi gian n');
ylabel('Bien do');
legend('x[n]','s[n]','y4[n]','y5[n]');

% sum of squared errors (SSE) bw output filtered signal & original signal (w/o noise)
% SSE lower is better
diff1 = y4(2:L-1) - s(1:L-2);
diff2 = y5(3:L) - s(1:L-2);
disp(sum(diff1.*diff1))
disp(sum(diff2.*diff2))

