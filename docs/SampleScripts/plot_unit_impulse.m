L1 = 10;
L2 = 20;
n = -L1:L2;% time-base vector
x = [zeros(1,L1), 1, zeros(1,L2)];% amplitude vector
stem(n,x,'fill')
title('Unit impulse');
xlabel('Time (n)');
ylabel('Amplitude');