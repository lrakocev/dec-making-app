function separation_across_tasks_by_subject_3d(tasks,tables,colors,save_to)
hs = [];
figure
legend_entry = [];
timings = [];
engs = [];
for i = 1:length(tasks)
    tit = tasks(i);
    table = tables(i);
    table = table{1};

    subjects = unique(table.subjectidnumber);
    legend_entry = [legend_entry; tit + ": n = " + string(length(subjects))];


    curr_mean_timing = [];
    curr_mean_eng = [];
    curr_mean_dec_eng = [];

    for s = 1:length(subjects)
        subid = subjects(s);
        curr_table = table(table.subjectidnumber == string(subid{1}), :);

        curr_table.decision_timing = get_arr(curr_table.decision_timing);
        curr_table = curr_table(curr_table.decision_timing ~= 0, :);
        
        timing = curr_table.decision_timing; 
        mean_timing = mean(timing, 'omitnan');
        curr_mean_timing = [curr_mean_timing; mean_timing];
    
        eng = get_arr(curr_table.avg_eng);
        mean_eng = mean(eng, 'omitnan');
        curr_mean_eng = [curr_mean_eng; mean_eng];

        dec_eng = get_arr(curr_table.eng_around_dec);
        mean_dec_eng = mean(dec_eng, 'omitnan');
        curr_mean_dec_eng = [curr_mean_dec_eng; mean_dec_eng];
        
    end

    mean_timing = mean(curr_mean_timing, 'omitnan');
    mean_eng = mean(curr_mean_eng, 'omitnan');
    mean_dec_eng = mean(curr_mean_dec_eng, 'omitnan');
    std_err_timing = std(curr_mean_timing, 'omitnan') / sum(~isnan(curr_mean_timing));
    std_err_eng = std(curr_mean_eng, 'omitnan') / sum(~isnan(curr_mean_eng));
    std_err_dec_eng = std(curr_mean_dec_eng, 'omitnan') / sum(~isnan(curr_mean_dec_eng));
    
    h = scatter3(mean_timing, mean_eng, mean_dec_eng, 20, colors(i), 'filled');
    hold on 
    plot3([mean_timing, mean_timing]', [-std_err_eng, std_err_eng]'+mean_eng', [mean_dec_eng, mean_dec_eng]', colors(i))
    hold on 
    plot3([-std_err_timing, std_err_timing]'+mean_timing', [mean_eng, mean_eng]',[mean_dec_eng, mean_dec_eng]', colors(i))
    hold on
    plot3([mean_timing, mean_timing]', [mean_eng, mean_eng]',[-std_err_dec_eng, std_err_dec_eng]'+mean_dec_eng', colors(i))
    hold on

    hs = [hs; h];
end
    

title("3D Separation of Eyetracking Data By Subject")
legend(hs, legend_entry);
xlabel('Timing')
ylabel('Avg Eng')
zlabel('Eng Around Decision')
fighandle = gcf;
set(gcf,'renderer','Painters')
saveas(fighandle,strcat(save_to,'3d_plot_by_subjects.fig'))

end