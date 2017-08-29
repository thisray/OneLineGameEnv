# OneLineGameEnv
a simple environment for RL experiment

* view_env: [view_env.ipynb](https://github.com/thisray/OneLineGameEnv/blob/master/view_env.ipynb)

'''

    from env.one_line_game import OneLineGameEnv

    env = OneLineGameEnv()

    env.start()

    state, reward, terminate, rets = env.step(<int_type_action>)

    state = env.reset()



    '''
    action list:
        0 :   'right',
        1 :   'down',
        2 :   'left',
        3 :   'up',
        4 :   'flag'
    '''

'''
