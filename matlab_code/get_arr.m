function c = get_arr(input)

b = convertCharsToStrings(input);
c = [];
for i = 1:size(b)
    if b(i) == ""
        b(i) = "0";
    end
    c = [c; str2num(b(i))];
end
end