"""Distributed Reinforcement Learning"""

from necklace.rpc import svctmp


class ServerXx(svctmp.TmplServer):
    def __init__(self, cfg={}):
        super(ServerXx, self).__init__(cfg)

        self.trainer_url = self.cfg.get('trainer_url')
        self.trainer_cli = self.mk_rpc_cli(self.trainer_url)

    def game_self_play(self):
        self.g_sleep(3)
        play_data = range(100)
        data_n = 100

        # one self-play
        self.q_put_state(1, play_data, data_n)

    def do_put_data(self, dt, dt_n, *args, **kwargs):

        r = self.do_rpc_call(
            self.trainer_cli, 'fn_put_data',
            data=dt, data_n=dt_n)

        # put one data
        self.q_put_state(0, r)

        return r

    def fn_put_data(self, msg, *args, **kwargs):

        m = self.msg_unpck(msg)
        ndid = m.get('ndid')
        role = m.get('role')
        rank = m.get('rank')

        # TODO: do check

        msg = m.get('msg') or {}
        dt = msg.get('data')
        dt_n = msg.get('data_n')

        # save the data
        #for d in dt:
        #    print(d)

        self.q_put_state(2, dt, dt_n)

        rep = self.mk_rpc_ret('fn_put_data', 'ok')

        return self.msg_pck(rep)


class ServiceXx(svctmp.TmplService):
    def __init__(self, cfg={}):
        super(ServiceXx, self).__init__(cfg)

        self.init_svr()

        self.cli_pool = {}

        self.q_put_state(0, 'start_self_play')

    def init_svr(self):
        self.cfg['q'] = self.q
        svr = ServerXx(self.cfg)
        self.svr = svr

    def check_state(self, *args, **kwargs):

        print('ServiceXx:  >>> check_state', self.svr.rank)

        #svr_state = self.svr.state

        s = self.q_get_state(timeout=1)
        svr_state, sa, skw = s

        print('ServiceXx:  >>> svr_state', svr_state)

        if svr_state == 0:
            self.svr.game_self_play()

        elif svr_state == 1:
            if sa:
                play_data, data_n = sa
                self.svr.do_put_data(play_data, data_n, self.svr.rank)
            else:
                pass

        elif svr_state == 2:
            if sa:
                play_data, data_n = sa
                for d in play_data:
                    print(d)
            else:
                pass

            self.q_put_state(0, 'start_self_play')

        else:
            pass

