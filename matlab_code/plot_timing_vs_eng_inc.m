function plot_timing_vs_eng_inc(table, tit, save_to)

table.decision_timing = get_arr(table.decision_timing);
table = table(table.decision_timing ~= 0, :);

timing = table.decision_timing;
num_e = get_arr(table.num_maxes);
[r,p] = corrcoef(timing, num_e);

xFit = 0:1:max(timing);
linearCoefficients = polyfit(timing, num_e, 1);

m = linearCoefficients(1);
c = linearCoefficients(2);

yFit = polyval(linearCoefficients, xFit);

figure
scatter(timing, num_e)
hold on
plot(xFit, yFit, 'b', 'LineWidth', 2);

xlabel('dec timing')
ylabel('num inc eng')
title(strcat(tit, " (by trial)", " slope: ", string(m), " r : ", string(r(2))))

savefig(strcat(save_to,strrep(tit, " ", "_"), "dec_timing_vs_num_inc_in_eng.fig"))
end