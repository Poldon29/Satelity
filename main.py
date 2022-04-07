from datetime import datetime, timedelta
from turtle import position
import numpy as np
from python.read_yuma import read_yuma
from python.date2tow import date2tow
from python.skyplot_one_epoch import plot_skyplot
import matplotlib.pyplot as mp

class dop:
    def __init__(self,amount):
        self.PDOP = []
        self.TDOP = []
        self.GDOP = []
        self.HDOP = []
        self.VDOP = []
        self.amount = amount
    def pass_DOPS(self,PDOP:int ,TDOP:int, GDOP:int, HDOP:int, VDOP:int):
        self.PDOP.append(PDOP)
        self.TDOP.append(TDOP)
        self.GDOP.append(GDOP)
        self.HDOP.append(HDOP)
        self.VDOP.append(VDOP)
    def draw_plot(self):
        
        y = np.array(self.PDOP)
        y1 = np.array(self.TDOP)
        y2 = np.array(self.GDOP)
        y3 = np.array(self.HDOP)
        y4 = np.array(self.VDOP)

        mp.plot(y)
        mp.plot(y1)
        mp.plot(y2)
        mp.plot(y3)
        mp.plot(y4)
        
        mp.show()
class Satellites:
    def __init__(self, file_name: str, start_date: datetime, minutes: int, mask: int, observer_pos: list):
        self.file = file_name
        self.naval = read_yuma(self.file)
        self.start_date = start_date
        self.end_date = self.start_date + timedelta(days=1)
        self.interval = timedelta(minutes=minutes)
        self.mask = mask
        self.r_neu = self.r_neu(observer_pos[0], observer_pos[1])

        # WGS84
        self.a = 6378137    #promien ziemi w m
        self.e2 = 0.00669438002290  #kwadrat pierwszego mimośrodu

    def datetime_to_list(self, date: datetime):
        date = [date.year,
                date.month,
                date.day,
                date.hour,
                date.minute,
                date.second]
        return date

    def satellite_xyz(self, week: int, tow: int, nav: np.ndarray):
        id, health, e, toa, i, omega_dot, sqrta, Omega, omega, m0, alfa, alfa1, gps_week = nav

        t = week * 7 * 86400 + tow
        toa_weeks = gps_week * 7 * 86400 + toa
        tk = t - toa_weeks

        """algorytm"""
        u = 3.986005 * (10 ** 14)
        omega_e = 7.2921151467 * (10 ** -5)
        a = sqrta ** 2
        n = np.sqrt(u / (a ** 3))
        Mk = m0 + n * tk

        E1 = Mk
        Ei = Mk + e * np.sin(E1)
        while np.abs(Ei - E1) >= (10 ** -12):
            E1 = Ei
            Ei = Mk + e * np.sin(E1)
            last_Ei = Ei
            #print(E1, E1, Ei - E1)
            if(last_Ei == Ei):
                break

        Ek = Ei
        vk = np.arctan2(np.sqrt(1 - e ** 2) * np.sin(Ek), np.cos(Ek) - e)

        phi_k = vk + omega
        rk = a * (1 - e * np.cos(Ek))
        xk = rk * np.cos(phi_k)
        yk = rk * np.sin(phi_k)
        omega_k = Omega + (omega_dot - omega_e) * tk - omega_e * toa

        Xk = xk * np.cos(omega_k) - yk * np.cos(i) * np.sin(omega_k)
        Yk = xk * np.sin(omega_k) + yk * np.cos(i) * np.cos(omega_k)
        Zk = yk * np.sin(i)

        return Xk, Yk, Zk          
    # Zamiana phi,lambda na x,y,z
    def phi_to_xyz(self, phi: float, lamda: float, height: float):
        a = 6378137
        e2 = 0.00669438002290
        phi = np.deg2rad(phi)
        lamda = np.deg2rad(lamda)
        N = a / (np.sqrt(1 - e2*(np.sin(phi)**2)))

        x = (N + height)*np.cos(phi)*np.cos(lamda)
        y = (N + height)*np.cos(phi)*np.sin(lamda)
        z = (N*(1-e2) + height)*np.sin(phi)
        return x, y, z
    # Macierz obrotu do układu horyzontalnego
    def r_neu(self, phi: float, lamda: float):
        phi = np.deg2rad(phi)
        lamda = np.deg2rad(lamda)
        matrix = np.array([[-np.sin(phi)*np.cos(lamda), -np.sin(lamda), np.cos(phi)*np.cos(lamda)],
                           [-np.sin(phi)*np.sin(lamda), np.cos(lamda), np.cos(phi)*np.sin(lamda)],
                           [np.cos(phi), 0, np.sin(phi)]])
        return matrix
    # Przeliczenie wektora satelita-odbiornik
    def neu(self, r_neu: np.array, Xsr: list):
        #print(r_neu) #wynik
        r_neu = np.transpose(r_neu)
        #print(r_neu) #wynik2
        return np.dot(r_neu, Xsr)
    #################### draw functions
    def draw_qt(self,qt_sat):
        ax = mp.subplots()
        hour_list = [x for x in range(25)]
        p1 = mp.bar(hour_list,qt_sat)
        mp.xticks(hour_list,hour_list)
        mp.yticks([x for x in range(max(qt_sat)+1)],[x for x in range(max(qt_sat)+1)])
        mp.show()
    def draw_elevation(self,az_list):
        fin_list = []
        for idx in range(self.naval.shape[0]):
            pom_list = []
            for el in az_list:
                pom_list.append(el[idx])
            fin_list.append(pom_list)
        date_list = [(self.start_date + timedelta(hours = x*4)).strftime("%H")+":00" for x in range(7)]
        x_list = []
        for i in range(7):
            x_list.append(i*16)
        x = np.array(x_list)

        mp.xticks(x,date_list)
        for el in fin_list:
            y_points = np.array(el)        
            mp.plot(y_points)
        mp.suptitle("Elewacja")
        mp.ylim(self.mask, 90)
        mp.show()
    ####################
    def satellites_coordinates(self):
        A = np.zeros((0, 4))
        number_of_satellites = self.naval.shape[0]
        era_date = self.start_date
        #dane do wykresow
        az_list = []
        qt_sat = []
        dops_lists = dop(self.naval.shape[0])
        ####
        positions = []
        while era_date <=  self.end_date:
            
            data = self.datetime_to_list(era_date)
            week, tow = date2tow(data)
            pom_list = []
            ### licznik widocznych
            ct = 0
            temp_date = era_date.strftime("%M")
            ####
            A = np.zeros((0,4))
            for id in range(number_of_satellites):
                nav = self.naval[id,:]
                Xs = self.satellite_xyz(week, tow, nav)  # xyz satelity
                Xr = self.phi_to_xyz(52, 21, 100)
                Xsr = [i - j for i, j in zip(Xs, Xr)]
                neu = self.neu(self.r_neu, Xsr)  # neu satelity
                n, e, u = neu
                Az = np.arctan2(e, n)  # arctan(e/n)
                Az = np.degrees(Az) #azymut
                if Az < 0:
                    Az += 360
                el = np.arcsin(u / (np.sqrt(n ** 2 + e ** 2 + u ** 2))) 
                el = np.degrees(el) #elewacja
                r = np.sqrt(Xsr[0] ** 2 + Xsr[1] ** 2 + Xsr[2] ** 2)

                if era_date == self.start_date:
                    positions_pom = [id+1,Az,el]
                    positions.append(positions_pom)

                if el > self.mask:
                    A1 = np.array([(-(Xs[0]-Xr[0]) / r),
                                (-(Xs[1] - Xr[1]) / r),
                                (-(Xs[2] - Xr[2]) / r),
                                1])
                    A = np.vstack([A, A1])
                    #### licz widocznych
                    ct+=1
                    ####
                    pom_list.append(el)         
                else:
                    pom_list.append(0)
            if(temp_date == "00"):
                qt_sat.append(ct)
            ###########
            az_list.append(tuple(pom_list))
            #############
             
            ##########
       
        
            Q = np.linalg.inv(np.dot(A.transpose(), A))
            qx, qy, qz, qt = Q.diagonal()
            Qxyz = Q[:3, :3]

            PDOP = np.sqrt(qx + qy + qz)
            TDOP = np.sqrt(qt)
            GDOP = np.sqrt(PDOP**2 + TDOP**2)

            Qneu = self.r_neu.transpose() @ Qxyz @ self.r_neu
            qn, qe, qu = Qneu.diagonal()
            HDOP = np.sqrt(qn + qe)
            VDOP = np.sqrt(qu)
            PDOPneu = np.sqrt(HDOP**2 + VDOP**2)
            era_date += self.interval
            dops_lists.pass_DOPS(PDOPneu,TDOP, GDOP, HDOP, VDOP)
        dops_lists.draw_plot()
        self.draw_qt(qt_sat)
        self.draw_elevation(az_list)
        plot_skyplot(positions)

if __name__ == "__main__":
    sat = Satellites(file_name='almanac.yuma.week0150.589824.txt', start_date=datetime(year=2022, month=2, day=25), minutes=15, mask=10, observer_pos=[51,22,100])
    #plot_skyplot()
    sat.satellites_coordinates()