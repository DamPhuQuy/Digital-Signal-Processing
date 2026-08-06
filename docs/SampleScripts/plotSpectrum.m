N = 1024; % so diem tren truc tan so w
w = -pi:pi/N:pi;

% sinh va ve do thi x[n] = (1/2)^n
n = 0:20;
x = (1/2).^n;
subplot(3,1,1);
stem(n,x,'fill');
title('x[n]');
% sinh va ve do thi X(e^jw)
X = freqz(x,1,w); % nhan duoc X(ejw)
modX = abs(X);
subplot(3,1,2);
plot(w,modX);grid;
axis([min(w), max(w) , 0, max(modX)]);
title ('Pho bien do tin hieu');
xlabel('Tan so w (radians)');
phaseX = angle(X);
subplot(3,1,3);
plot(w,phaseX);grid;
axis([min(w), max(w) , min(phaseX), max(phaseX)]);
title ('Pho pha tin hieu');
xlabel('Tan so w (radians)');