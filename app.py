import asyncio
from turtle import Turtle
import socketio

from aiortc import (
    RTCIceCandidate,
    RTCPeerConnection,
    RTCSessionDescription,
    VideoStreamTrack,
    RTCIceServer,
    RTCConfiguration,
)
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder, MyVideoHandler

from aiortc import sdp

import cv2
import numpy as np

rtcIceServer = RTCIceServer("stun:stun.l.google.com:19302")
configuration = RTCConfiguration()
configuration.iceServers = [rtcIceServer]
peerConnection = None


async def handle_stream(recorder):
    # sio = socketio.Client(logger=True)
    sio = socketio.AsyncClient(logger=True)

    @sio.event
    async def connect():
        print('handle_connect')
        await sio.emit('watcher')
    
    @sio.on('offer')
    async def handle_offer(sid, data):
        # print('handle_offer', sid, data)
        print('handle_offer', sid)
        global peerConnection 
        peerConnection= RTCPeerConnection(configuration)
        
        @peerConnection.on("track")
        def on_track(track):
            print("Receiving %s" % track.kind)
            recorder.addTrack(track)
        
        # @peerConnection.on("icecandidate")
        # def on_icecandicate(event):
        #     # sio.emit()

        await peerConnection.setRemoteDescription(RTCSessionDescription(data['sdp'], data['type']))
        await recorder.start()
        print('record has been started')
        answer = await peerConnection.createAnswer()
        await peerConnection.setLocalDescription(answer)
        answer_dict = {
            'sdp' : peerConnection.localDescription.sdp,
            'type' : peerConnection.localDescription.type
        }
        await sio.emit('answer', (sid, answer_dict))
                
    @sio.on('candidate')
    async def handle_candidate(sid, data):
        print('4 handle_candidate', sid, data)
        candidate = sdp.candidate_from_sdp(data['candidate'])
        candidate.sdpMid = data['sdpMid']
        candidate.sdpMLineIndex = data['sdpMLineIndex']
        if peerConnection is not None:
            await peerConnection.addIceCandidate(candidate)
        # print(peerConnection.localDescription)
        # await sio.emit('candidate', (sid, data))

    @sio.on('broadcaster')
    async def handle_broadcaster():
        print('handle_broadcaster')
        await sio.emit('watcher')
    
    await sio.connect('http://localhost:9000')
    print('my sid is', sio.sid)

    await sio.wait()

import takepicturee
import check_face
from os import makedirs
from os import listdir
from os.path import isdir, isfile, join
import os
from os import system
from charset_normalizer import models
import face_recognition
import face_recognition_models
import time
from contextlib import nullcontext
from cProfile import run
from unittest import result
import example2
import posturedetect
import recognitioneye
import requests


def upgrade():
    consequence = check_face.trains()
    return consequence


async def show_stream(recorder):

    while True:
        
        version = int(input("1. 얼굴인식, 2. 공간인식, 3. 자세교정, 4. 취침판단: "))

        if(version == 1):
            models = upgrade()
            sum = 0
            change = 10
            while True:
                frame = recorder.read()
                if frame is not None:
                    result = check_face.run(models, frame, change)
                    print(result)
                    if result == 10:
                        sum += 1
                        if sum >= 5:
                            change = 100
                        else:
                            change = 10
                    elif result == 100:
                        continue
                    elif (type(result) == str):
                        idcheck = {
                            'username' : result
                        }
                        res = requests.post('http://localhost:9999/User',json = idcheck)
                        print(res.status_code)
                        print(res.json())

                        break
                    np.save('frame.npy', frame)
                await asyncio.sleep(2)
        
        elif(version == 2):
            a = example2.modelready()
            num1 = 0
            num2 = 0
            while True:
                frame = recorder.read()
                if frame is not None:
                    # print(frame.shape)
                    proto = example2.newdata(frame)
                    result = example2.predict(proto, a)
                    print(result)
                    if(result == 0):
                        num1 += 1
                        if(num1 >= 5):
                            print('제 1공간')
                    
                    if(result == 1):
                        num2 += 1
                        if(num2 >= 5):
                            print('제 2공간')
                    np.save('frame.npy', frame)

                await asyncio.sleep(2)
        
        elif(version == 3):
            sum = 0
            while True:
                frame = recorder.read()
                if frame is not None:
                    result =  posturedetect.production(frame)
                    sum += 1
                    posturecheck = {
                            'statement' : result
                    }
                    res = requests.post('http://localhost:9999/User',json = posturecheck)
                    print(res.status_code)
                    print(res.json())
                    if sum > 10:
                        break
                    
                np.save('frame.npy', frame)

                await asyncio.sleep(2)
        
        elif(version == 4):

            EYE_CLOSED_COUNTER = 0

            while True:
                frame = recorder.read()
                if frame is not None:
                    result = recognitioneye.sleepdetect(frame, EYE_CLOSED_COUNTER)
                    print(result)
                    if(result == 100):
                        print(EYE_CLOSED_COUNTER)
                        EYE_CLOSED_COUNTER += 1
                    if(type(result) == str):
                        eye_state = {
                            'sleepornot' : result
                        }
                        res = requests.post('http://localhost:9999/User',json = eye_state)
                        print(res.json())
                        break

                np.save('frame.npy', frame)

                await asyncio.sleep(2)
        await asyncio.sleep(2)
            

async def main(recorder):
    task1 = loop.create_task(handle_stream(recorder))
    task2 = loop.create_task(show_stream(recorder))
    await asyncio.wait([task1, task2])

if __name__ == '__main__':

    # recorder = MediaRecorder('video.mp4')
    recorder = MyVideoHandler('video.mp4')
    
    # run event loop
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(
            main(
                recorder
            )
        )

    except KeyboardInterrupt:
        pass

    finally:
        # cleanup
        loop.run_until_complete(recorder.stop())
        loop.run_until_complete(peerConnection.close())
        cv2.destroyAllWindows()   