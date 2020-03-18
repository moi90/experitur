%.gif: %.py
	experitur clean --all -y $<
	tools/record.exp $<
	ttyrec2gif -in ttyrecord -out $@