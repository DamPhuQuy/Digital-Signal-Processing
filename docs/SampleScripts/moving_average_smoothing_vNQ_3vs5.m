% chuong trinh lam tron 1 tin hieu de khu nhieu (denoise)
% so sanh hieu ung lam tron cua 2 bo loc trung binh truot N diem voi N=3 va N=5
% Bai toan: ta co tin hieu bi lan voi nhieu x[n], di khoi phuc lai
% tin hieu goc s[n] (an trong x[n])
% Ky vong KQ: y[n] cang giong s[n] cang tot

% sinh tin hieu bi lan voi nhieu cong (additive noise)
clear
close all                        % close all figures
A = 0.5;                        % cong suat nhieu ti le voi sqr(A)  
L = 100;                         % do dai tin hieu
n = 0:L-1;                      % bien thoi gian roi rac
d = A*randn(1,L);             % sinh tin hieu Gaussian white noise d[n] (A=std cua tin hieu nhieu)
s = 2*n.*(0.9.^n);              % sinh tin hieu goc s[n] = 2n(0.9)^n
x = s + d;                      % tin hieu co nhieu x[n]=s[n]+d[n]

% cai dat bo loc voi kich thuoc N=3 co PTSP: y[n] = 1/3*(x[n]+x[n-1]+x[n-2]) 
b = 1/3*ones(1,3);
a = [1];
y3 = filter(b,a,x);

% cai dat bo loc voi kich thuoc N=5 co PTSP: y[n] = 1/5*(x[n]+x[n-1]+x[n-2]+x[n-3]+x[n-4]) 
b = 1/5*ones(1,5);
a = [1];
y5 = filter(b,a,x);

% ve do thi xep chong x[n], s[n] va y3[n] va y5[n] de test ket qua
hold all
plot(n(1:L-2),x(1:L-2),'b-');
plot(n(1:L-2),s(1:L-2),'g-');
plot(n(1:L-2),y3(2:L-1),'k+-'); % y3[n] bi dich phai 1 mau (tre pha) so voi x[n]
plot(n(1:L-2),y5(3:L),'r.-');       % y5[n] bi dich phai 2 mau (tre pha) so voi x[n]
hold off
xlabel('Chi so mau n');
ylabel('Bien do');
legend('x[n]','s[n]','y3[n]','y5[n]');

% sum of squared errors (SSE) bw output signal & original signal (w/o noise)
% SSE lower is better
diff1 = y3(2:L-1) - s(1:L-2);
diff2 = y5(3:L) - s(1:L-2);
disp(sum(diff1.*diff1))
disp(sum(diff2.*diff2))

% co the tang N de lam tron tin hieu nhieu hon (khu duoc noise nhieu hon),
% nhung can tranh over-smoothing (lam tron qua nhieu dan den meo tin hieu) khi
% N qua lon

