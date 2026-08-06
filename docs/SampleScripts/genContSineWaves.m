% generate sampled copies of continuous-time sine waves with different frequencies
Fs = 1e3; % sampling frequency
D = 1.2;  % duration (in seconds)
t = 0:1/Fs:D; % create time vector t from 0(s) to D (s)
clf;

% create sine functions with different freqencies

F = 0;
x = cos(2*pi*F*t); 
subplot(5,1,1);
plot(t, x);
title('F=0');

F = 10;
x = cos(2*pi*F*t); 
subplot(5,1,2);
plot(t, x);
title('F=10');

F = 20;
x = cos(2*pi*F*t); 
subplot(5,1,3);
plot(t, x);
title('F=20');

F = 100;
x = cos(2*pi*F*t); 
subplot(5,1,4);
plot(t, x);
title('F=100');

F = 200;
x = cos(2*pi*F*t); 
subplot(5,1,5);
plot(t, x);
title('F=200');
