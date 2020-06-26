%.gif %.txt %.csv: %.py
	experitur clean --all -y $<
	tools/record.exp $< | tee $(<:.py=.txt)
	ttyrec2gif -in ttyrecord -out $(<:.py=.gif)
	experitur collect $< $(<:.py=.csv)