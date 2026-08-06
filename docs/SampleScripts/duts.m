% plot freq response, given its expression
w = -pi:pi/100:pi;          % vecto tan so (truc t/s)
H = 1/3*(1 + exp(-j*w) + exp(j*w));   % dap ung tan so
Habs = abs(H);              % dap ung bien do
Hphase = angle(H);          % dap ung pha

figure(1)
subplot(2,1,1);
plot(w/pi,Habs);
grid;
title('Dap ung bien do');

subplot(2,1,2);
plot(w/pi,Hphase);
grid;
title('Dap ung pha');

