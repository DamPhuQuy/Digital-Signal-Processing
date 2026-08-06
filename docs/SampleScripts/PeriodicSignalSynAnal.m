% CT nay lam 2 viec:
% - tong hop tin hieu tuan hoan x[n] tu N tin hieu sin thuc (song hai)
%   co tan so cach deu nhau trong dai [0,2pi) va phat ra loa
% - phan tich pho cua tin hieu vua moi tong hop duoc
% Chu y: moi lan chay CT ket qua se thay doi, co the cong them nhieu trang
% vao tin hieu va quan sat ket qua
n = 0:16000; % vecto thoi gian roi rac
Fs = 16000;  % tan so lay mau

% tong hop tin hieu tuan hoan x[n] tu cac song hai
N = 20; % so luong tin hieu sin/song hai (chinh la chu ky cua x[n])
x = zeros(1,length(n)); % tin hieu x[n] tong hop tu N tin hieu sin
for k = 0:N-1
    A = randi(N); % sinh bien do ngau nhien
    x = x + A*cos((2*pi*k/N)*n); % w(k) = 2*pi*k/N
end
%x = x + 3*randn(1, length(x)); % them nhieu trang (white noise) vao tin hieu
% ve do thi tin hieu
subplot(3,1,1);
stem(n(1:100),x(1:100),'fill'); % chi ve 100 mau dau tien
title(['Tin hieu x[n] tuan hoan voi chu ky N=', num2str(N)])
sound(x,Fs); % khoi phuc TH voi ts lay mau Fs
pause(3);

% phan tich pho bien do cua tin hieu vua duoc sinh ra x[n]
% ve do thi |X(e^jw)| va |X(F)|
N = 1024; % so diem tren truc tan so w
w = -pi:pi/N:pi;
X = freqz(x,1,w); % X(e^jw) la vecto phuc theo w
% ve do thi pho bien do |X(e^jw)| cua tin hieu
subplot(3,1,2);
plot(w,abs(X)); 
grid; 
axis([min(w), max(w), 0, max(abs(X))]);
title ('Pho bien do tin hieu tren tan so w va thang tuyen tinh');
xlabel('w(radians)');
% ve do thi pho bien do |X(F)| cua tin hieu
F = Fs * w / (2*pi); % chuyen tu t/s ly thuyet w (radian) sang t/s thuc F (Hz)
modX = abs(X)/Fs;
logXF = 20*log10(modX); % lay log10 cua |X(F)| de nen dai bien thien
subplot(3,1,3);
plot(F,logXF); 
grid; 
axis([0, Fs/2 , min(logXF), max(logXF)]);
title ('Pho bien do tin hieu tren tan so F va thang logarit');
xlabel('F(Hz)');
ylabel('Magnitude (dB)');

