Developer Documentation
=======================

How to record example runs with :code:`ttyrec`
----------------------------------------------

.. code-block:: sh

    cd examples
    ttyrec -e ./minimalshell
    # Type commands...
    # Ctrl-d
    ttyrec2gif -in ttyrecord -out <fn>.gif



Requirements
~~~~~~~~~~~~

- ttyrec
- ttyrec2gif: :code:`go get github.com/sugyan/ttyrec2gif`