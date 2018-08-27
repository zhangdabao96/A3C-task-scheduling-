
import multiprocessing
import threading
import tensorflow as tf
import numpy as np
#import gym
import os
import random
#import shutil
import matplotlib.pyplot as plt
from TaskDeal import task


N_WORKERS = multiprocessing.cpu_count() 
MAX_EP = 1000

GLOBAL_NET_SCOPE = 'Global_Net'
UPDATE_GLOBAL_ITER =10  
GAMMA = 0.9
ENTROPY_BETA = 0.001
LR_A = 0.001    # learning rate for actor
LR_C = 0.001    # learning rate for critic
GLOBAL_RUNNING_R = []
TIME=[]
GLOBAL_EP = 0
#C_COST=[]
All_COST=[]
load=[]
totalcost=[]
#env = gym.make(GAME)
#N_S = env.observation_space.shape[0] 
N_S =85
#N_A = env.action_space.n
N_A_L=10
N_A_N=5


class ACNet(object):
    def __init__(self, scope, globalAC=None):

        if scope == GLOBAL_NET_SCOPE:   # get global network
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
                #self.a_params, self.c_params = self._build_net(scope)[-2:]
                self.all_params = self._build_net(scope)[-1]
        else:   # local net, calculate losses
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
                self.a_his = tf.placeholder(tf.int32, [None,N_A_N ], 'A')
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')

                #self.a_prob, self.v, self.a_params, self.c_params = self._build_net(scope)
                self.a0_prob, self.a1_prob, self.a2_prob, self.a3_prob, self.a4_prob,  self.v, self.all_params = self._build_net(scope)
                td = tf.subtract(self.v_target, self.v, name='TD_error')
                
                # 接着计算 critic loss 和 actor loss
                with tf.name_scope('c_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td))
                                
                with tf.name_scope('a_loss'):
                    '''
                    log_prob = tf.reduce_sum(tf.log(self.a_prob) * tf.one_hot(self.a_his, N_A, dtype=tf.float32), axis=1, keep_dims=True)
                    exp_v = log_prob * tf.stop_gradient(td)
                    entropy = -tf.reduce_sum(self.a_prob * tf.log(self.a_prob + 1e-5),
                                             axis=1, keep_dims=True)  # encourage exploration
                    self.exp_v = ENTROPY_BETA * entropy + exp_v
                    self.a_loss = tf.reduce_mean(-self.exp_v)
                    '''
                    
                    #ACTION0
                    log_prob0 = tf.reduce_sum(tf.log(self.a0_prob) * tf.one_hot(self.a_his[:,0], N_A_L, dtype=tf.float32), axis=1, keep_dims=True)
                    exp_v0 = log_prob0 * tf.stop_gradient(td)
                    entropy0 = -tf.reduce_sum(self.a0_prob * tf.log(self.a0_prob + 1e-5),
                                             axis=1, keep_dims=True)  # encourage exploration
                    self.exp_v0 = ENTROPY_BETA * entropy0 + exp_v0
                    self.a0_loss = tf.reduce_mean(-self.exp_v0)
                    
                    #ACTION1
                    log_prob1 = tf.reduce_sum(tf.log(self.a1_prob) * tf.one_hot(self.a_his[:,1], N_A_L, dtype=tf.float32), axis=1, keep_dims=True)
                    exp_v1 = log_prob1 * tf.stop_gradient(td)
                    entropy1 = -tf.reduce_sum(self.a1_prob * tf.log(self.a1_prob + 1e-5),
                                             axis=1, keep_dims=True)  # encourage exploration
                    self.exp_v1 = ENTROPY_BETA * entropy1 + exp_v1
                    self.a1_loss = tf.reduce_mean(-self.exp_v1)
                    
                    #ACTION2
                    log_prob2 = tf.reduce_sum(tf.log(self.a2_prob) * tf.one_hot(self.a_his[:,2], N_A_L, dtype=tf.float32), axis=1, keep_dims=True)
                    exp_v2 = log_prob2 * tf.stop_gradient(td)
                    entropy2 = -tf.reduce_sum(self.a2_prob * tf.log(self.a2_prob + 1e-5),
                                             axis=1, keep_dims=True)  # encourage exploration
                    self.exp_v2 = ENTROPY_BETA * entropy2 + exp_v2
                    self.a2_loss = tf.reduce_mean(-self.exp_v2)
                    
                    #ACTION3
                    log_prob3 = tf.reduce_sum(tf.log(self.a3_prob) * tf.one_hot(self.a_his[:,3], N_A_L, dtype=tf.float32), axis=1, keep_dims=True)
                    exp_v3 = log_prob3 * tf.stop_gradient(td)
                    entropy3 = -tf.reduce_sum(self.a3_prob * tf.log(self.a3_prob + 1e-5),
                                             axis=1, keep_dims=True)  # encourage exploration
                    self.exp_v3 = ENTROPY_BETA * entropy3 + exp_v3
                    self.a3_loss = tf.reduce_mean(-self.exp_v3)

                    #ACTION4
                    log_prob4 = tf.reduce_sum(tf.log(self.a4_prob) * tf.one_hot(self.a_his[:,4], N_A_L, dtype=tf.float32), axis=1, keep_dims=True)
                    exp_v4 = log_prob4 * tf.stop_gradient(td)
                    entropy4 = -tf.reduce_sum(self.a4_prob * tf.log(self.a4_prob + 1e-5),
                                             axis=1, keep_dims=True)  # encourage exploration
                    self.exp_v4 = ENTROPY_BETA * entropy4 + exp_v4
                    self.a4_loss = tf.reduce_mean(-self.exp_v4)

                    self.a_loss=self.a0_loss+self.a1_loss+self.a2_loss+self.a3_loss+self.a4_loss



                with tf.name_scope('all_loss'):
                    self.all_loss=self.a_loss+self.c_loss


                # 用这两个 loss 计算要推送的 gradients
                with tf.name_scope('local_grad'):
                    #self.a_grads = tf.gradients(self.a_loss, self.a_params)
                    #self.c_grads = tf.gradients(self.c_loss, self.c_params)
                    self.all_grads = tf.gradients(self.all_loss, self.all_params)



            with tf.name_scope('sync'):  #同步
                with tf.name_scope('pull'): # 更新去 global
                    #self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalAC.a_params)]
                    #self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC.c_params)]
                    self.pull_all_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.all_params, globalAC.all_params)]


                with tf.name_scope('push'):  # 获取 global 参数
                    #self.update_a_op = OPT_A.apply_gradients(zip(self.a_grads, globalAC.a_params))
                    #self.update_c_op = OPT_C.apply_gradients(zip(self.c_grads, globalAC.c_params))
                    self.update_all_op = OPT_C.apply_gradients(zip(self.all_grads, globalAC.all_params))



    def _build_net(self, scope):
        w_init = tf.random_normal_initializer(0., .1)
        '''
        with tf.variable_scope('actor'):
            l_a = tf.layers.dense(self.s, 200, tf.nn.relu6, kernel_initializer=w_init, name='la')
            a_prob = tf.layers.dense(l_a, N_A, tf.nn.softmax, kernel_initializer=w_init, name='ap')
        with tf.variable_scope('critic'):
            l_c = tf.layers.dense(self.s, 100, tf.nn.relu6, kernel_initializer=w_init, name='lc')
            v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')  # state value
        '''
        with tf.variable_scope('actor_critic'):
            l_1 = tf.layers.dense(self.s, 64, tf.nn.relu6, kernel_initializer=w_init, name='l1')
            l_2 = tf.layers.dense(l_1, 64, tf.nn.relu6, kernel_initializer=w_init, name='l2')
            l_3 = tf.layers.dense(l_2, 64, tf.nn.relu6, kernel_initializer=w_init, name='l3')
            
            a0_prob = tf.layers.dense(l_3, N_A_L, tf.nn.softmax, kernel_initializer=w_init, name='a0_p')
            a1_prob = tf.layers.dense(l_3, N_A_L, tf.nn.softmax, kernel_initializer=w_init, name='a1_p')
            a2_prob = tf.layers.dense(l_3, N_A_L, tf.nn.softmax, kernel_initializer=w_init, name='a2_p')
            a3_prob = tf.layers.dense(l_3, N_A_L, tf.nn.softmax, kernel_initializer=w_init, name='a3_p')
            a4_prob = tf.layers.dense(l_3, N_A_L, tf.nn.softmax, kernel_initializer=w_init, name='a4_p')
        

            v = tf.layers.dense(l_3, 1, kernel_initializer=w_init, name='v')  # state value

        #a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
        #c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        all_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor_critic')
        #return a_prob, v, a_params, c_params
        return a0_prob,a1_prob, a2_prob, a3_prob, a4_prob, v, all_params
        

    def update_global(self, feed_dict):  # run by a local
        #actor_cost,critic_cost=SESS.run([self.a_loss, self.c_loss], feed_dict)
        actor_critic_cost=SESS.run([self.all_loss], feed_dict)

        print('actor_critic_cost:',actor_critic_cost)
        All_COST.append(actor_critic_cost)
        #C_COST.append(critic_cost)

        #SESS.run([self.update_a_op, self.update_c_op], feed_dict)  # local grads applies to global net
        SESS.run([self.update_all_op], feed_dict)  # local grads applies to global net

    def pull_global(self):  # run by a local
        #SESS.run([self.pull_a_params_op, self.pull_c_params_op])
        SESS.run([self.pull_all_params_op])

    def choose_action(self, s):  # run by a local
        
        prob_weights_0, prob_weights_1, prob_weights_2, prob_weights_3, prob_weights_4, = SESS.run([self.a0_prob,self.a1_prob,self.a2_prob,self.a3_prob,self.a4_prob], feed_dict={self.s: s[np.newaxis, :]}) #(1,85)        
        
        action=np.zeros(5,dtype=int)
        action[0] = np.random.choice(range(prob_weights_0.shape[1]),
                                  p=prob_weights_0.ravel())  # select action w.r.t the actions prob  
        action[1] = np.random.choice(range(prob_weights_1.shape[1]),
                                  p=prob_weights_1.ravel())  # select action w.r.t the actions prob  
        action[2] = np.random.choice(range(prob_weights_2.shape[1]),
                                  p=prob_weights_2.ravel())  # select action w.r.t the actions prob          
        action[3] = np.random.choice(range(prob_weights_3.shape[1]),
                                  p=prob_weights_3.ravel())  # select action w.r.t the actions prob          
        action[4] = np.random.choice(range(prob_weights_4.shape[1]),
                                  p=prob_weights_4.ravel())  # select action w.r.t the actions prob          


        #print(action)
        return action


class Worker(object):
    def __init__(self, name, globalAC):
        #self.env = gym.make(GAME).unwrapped #！！
        self.TaskGroup = task()
        
        self.name = name
        self.AC = ACNet(name, globalAC)  #name:W_1

    def work(self):
        global GLOBAL_RUNNING_R, GLOBAL_EP #：[],0
        total_step = 1
        buffer_s, buffer_a, buffer_r = [], [], []
        #TaskGroup = task()
        
        while not COORD.should_stop() and total_step < MAX_EP:
            
            #s = self.env.reset()#！！
            s = self.TaskGroup.TaskState()  #
            

            
            ep_r = 0
            while total_step < MAX_EP:
                # if self.name == 'W_0':
                #     self.env.render()
                
                action = self.AC.choose_action(s)
                
                #action=np.zeros(N_A)
                
                #action[a] = 1                
                
                s_, r,tmp_load,tmp_totalcost =self.TaskGroup.task_step(action)  #done??
                
                GLOBAL_RUNNING_R.append(r)
                TIME.append(10-r)
                load.append(tmp_load)
                totalcost.append(tmp_totalcost)

                ep_r += r
                buffer_s.append(s)
                buffer_a.append(action)
                buffer_r.append(r)
                #print(self.name,': total_step',total_step ,': ',r)
                
                #if total_step % UPDATE_GLOBAL_ITER == 0 or done:   # update global and assign to local net
                if total_step % UPDATE_GLOBAL_ITER == 0 :
                    #if done:
                    #    v_s_ = 0   # terminal
                    #else:
                    v_s_ = SESS.run(self.AC.v, {self.AC.s: s_[np.newaxis, :]})[0, 0]
                    
                    buffer_v_target = []
                    for r in buffer_r[::-1]:    # reverse buffer r
                        v_s_ = r + GAMMA * v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse() #反向列表中元素

                    buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.array(buffer_a), np.vstack(buffer_v_target)
                    feed_dict = {
                        self.AC.s: buffer_s,
                        self.AC.a_his: buffer_a,
                        self.AC.v_target: buffer_v_target,
                    }
                    self.AC.update_global(feed_dict)

                    buffer_s, buffer_a, buffer_r = [], [], []
                    self.AC.pull_global()

                s = s_
                
                total_step += 1
                print(total_step)
                '''
                if done:
                    if len(GLOBAL_RUNNING_R) == 0:  # record running episode reward
                        GLOBAL_RUNNING_R.append(ep_r)
                    else:
                        GLOBAL_RUNNING_R.append(0.99 * GLOBAL_RUNNING_R[-1] + 0.01 * ep_r)
                    print(
                        self.name,
                        "Ep:", GLOBAL_EP,
                        "| Ep_r: %i" % GLOBAL_RUNNING_R[-1],
                          )
                    GLOBAL_EP += 1
                    break
                '''


if __name__ == "__main__":
    

    SESS = tf.Session()

    with tf.device("/cpu:0"): #
        OPT_A = tf.train.RMSPropOptimizer(LR_A, name='RMSPropA')
        OPT_C = tf.train.RMSPropOptimizer(LR_C, name='RMSPropC')
        GLOBAL_AC = ACNet(GLOBAL_NET_SCOPE)  # we only need its params
        workers = []
        # Create worker
        for i in range(N_WORKERS):
            i_name = 'W_%i' % i   # worker name
            workers.append(Worker(i_name, GLOBAL_AC))
    COORD = tf.train.Coordinator()  #
    SESS.run(tf.global_variables_initializer())

    worker_threads = []
    for worker in workers:
        job = lambda: worker.work()
        t = threading.Thread(target=job)
        t.start()
        worker_threads.append(t)
    COORD.join(worker_threads)


    plt.plot(np.arange(len(TIME)), TIME)
    plt.xlabel('step')
    plt.ylabel('TIME')
    plt.show()

    plt.plot(np.arange(len(load)), load)
    plt.xlabel('step')
    plt.ylabel('load')
    plt.show()

    plt.plot(np.arange(len(totalcost)), totalcost)
    plt.xlabel('step')
    plt.ylabel('totalcost')
    plt.show()

    plt.plot(np.arange(len(All_COST)), All_COST)
    plt.xlabel('step')
    plt.ylabel('All_COST')
    plt.show()    

    print('time_mean:',np.mean(TIME[3000:3500]))
    print('load_mean:',np.mean(load[3000:3500]))
    print('totalcost_mean:',np.mean(totalcost[3000:3500]))

    with open('E:/网络智能/任务调度相关工作/并行改进A3C尝试/data/TIME.txt','w') as f: 
        for a in TIME:
           f.write(str(a)+'\n') 

    with open('E:/网络智能/任务调度相关工作/并行改进A3C尝试/data/load.txt','w') as f: 
        for a in load:
           f.write(str(a)+'\n')
    with open('E:/网络智能/任务调度相关工作/并行改进A3C尝试/data/totalcost.txt','w') as f: 
        for a in totalcost:    
           f.write(str(a)+'\n')
