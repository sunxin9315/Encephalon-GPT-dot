"""Independent, synthetic CPU-only foundation adversarial regression tests.

Run from repository root with:
  python3 -m unittest discover -s tests -p 'test_independent_audit.py' -v
No model, network, microphone, camera or GPU is used.
"""
from dataclasses import FrozenInstanceError, replace
import inspect
import json
import random
import unittest

from encephalon.contracts import (AsyncRequest, AsyncResult, Claim, ContractError,
    Header, Observation, Scope, Validity)
from encephalon.clock import ReplayClock
from encephalon.observations import ObservationBuffer
from encephalon.async_requests import RequestRegistry
from encephalon.memory import MemoryEvent, SessionMemory
from encephalon.playback import OutputPacket, PlaybackLedger, SpeechSpan
from encephalon.runtime import SessionRuntime

SCOPE = Scope('persona_A', 'v1', 'session_A')

def header(*, time=0, available=None, state=0, revision=0, request=None, scope=SCOPE,
           status='valid', reason=None, sources=('source',), provenance=()):
    available = time if available is None else available
    return Header(scope.persona_id, scope.persona_version, scope.session_id, state,
        time,time,available,sources,provenance,Validity(status,1,available-time,reason),revision,request)

def observation(*, revision=0, available=0, value=0, scope=SCOPE, status='valid', reason=None):
    return Observation(header(available=available,revision=revision,scope=scope,status=status,reason=reason),
                       'window', 'speech_semantic', {'value':value})

def packet(seq=0, *, available=0, scope=SCOPE, utterance='u'):
    return OutputPacket(header(time=available,scope=scope),utterance,seq,seq*80,(seq+1)*80,
                        (SpeechSpan(seq*80,(seq+1)*80,f'text_{seq}',(f'c{seq}',)),))

def ack(ledger, until, *, now, utterance=None):
    kwargs={'scope':ledger.scope,'now_ms':now}
    # API now being hardened by implementation worker. Never omit identity if supported.
    if 'utterance_id' in inspect.signature(ledger.acknowledge_played).parameters:
        kwargs['utterance_id']=ledger.utterance_id if utterance is None else utterance
    return ledger.acknowledge_played(until,**kwargs)

def stop(ledger, until, *, now):
    ledger.request_stop(scope=ledger.scope,now_ms=now)
    kwargs={'scope':ledger.scope,'now_ms':now}
    if 'utterance_id' in inspect.signature(ledger.confirm_stopped).parameters:
        kwargs['utterance_id']=ledger.utterance_id
    return ledger.confirm_stopped(until,**kwargs)

def registered(*, premises=None, dispatch=0, expires=1000):
    premises={} if premises is None else premises
    reg=RequestRegistry(SCOPE)
    req=AsyncRequest(header(request='request'),0,0,('claim',),premises,expires)
    reg.register(req,now_ms=0,state_version=0,question_version=0,
                 premise_versions=premises,active_claim_ids={'claim'})
    reg.dispatch('request',dispatch)
    return reg,req

def result(req, *, available=10, scope=SCOPE, **changes):
    values=dict(header=header(time=available,request='request',scope=scope),request_version=0,
                question_version=0,claim_ids=('claim',),premise_versions=req.premise_versions,
                valid_until_ms=req.valid_until_ms,payload={'answer':'synthetic only'})
    values.update(changes)
    return AsyncResult(**values)

def receive(reg,res,*,now=10,state=0,question=0,premises=None,claims=None):
    return reg.receive(res,now_ms=now,state_version=state,question_version=question,
        premise_versions={} if premises is None else premises,active_claim_ids={'claim'} if claims is None else claims)

class IndependentContracts(unittest.TestCase):
    def test_nested_payload_is_detached_and_immutable(self):
        original={'a':[{'b':1}]}
        obs=Observation(header(),'id','text',original)
        original['a'][0]['b']=9
        self.assertEqual(obs.payload['a'][0]['b'],1)
        with self.assertRaises(TypeError): obs.payload['a'][0]['b']=9
        with self.assertRaises(FrozenInstanceError): obs.observation_id='other'
        copy=obs.to_dict(); copy['payload']['a'][0]['b']=10
        self.assertEqual(obs.payload['a'][0]['b'],1)

    def test_nonfinite_json_and_bool_time_are_rejected(self):
        for bad in (float('nan'),float('inf'),b'bytes',object()):
            with self.subTest(bad=repr(bad)), self.assertRaises(ContractError):
                Observation(header(),'id','text',{'value':bad})
        for bad in (-1, True, 1.5, '3'):
            with self.subTest(time=bad), self.assertRaises(ContractError): replace(header(),event_start_ms=bad)

    def test_roundtrip_strict_fields_and_bad_age(self):
        obs=observation(available=50,value='ok')
        self.assertEqual(Observation.from_dict(obs.to_dict()).to_dict(),obs.to_dict())
        data=obs.to_dict(); data['future_label']='no'
        with self.assertRaises(ContractError): Observation.from_dict(data)
        with self.assertRaises(ContractError): replace(obs.header,validity=Validity('valid',1,0))

    def test_silence_is_not_missing(self):
        valid=Observation(header(),'silence','audio',{'energy':0})
        missing=Observation(header(status='missing',reason='packet_loss'),'missing','audio',{})
        self.assertNotEqual(valid.header.validity.status,missing.header.validity.status)

    def test_replay_clock_never_rewinds(self):
        c=ReplayClock(); c.advance_to(50)
        with self.assertRaises(ContractError): c.advance_to(49)
        self.assertEqual(c.now_ms,50)

class IndependentObservations(unittest.TestCase):
    def test_future_revision_does_not_shadow_visible_old(self):
        b=ObservationBuffer(SCOPE)
        b.add(observation(revision=1,available=100,value='future'))
        b.add(observation(revision=0,available=10,value='present'))
        self.assertEqual(b.visible(9),())
        self.assertEqual(b.visible(10)[0].payload['value'],'present')
        self.assertEqual(b.visible(99)[0].header.revision,0)
        self.assertEqual(b.visible(100)[0].header.revision,1)

    def test_seeded_permutations_preserve_causal_revision(self):
        for seed in range(30):
            revisions=[observation(revision=i,available=i*10,value=i) for i in range(8)]
            random.Random(seed).shuffle(revisions)
            b=ObservationBuffer(SCOPE)
            for obs in revisions: b.add(obs)
            for t in range(80): self.assertEqual(b.visible(t)[0].payload['value'],min(t//10,7))

    def test_duplicate_conflict_and_cross_scope(self):
        b=ObservationBuffer(SCOPE); obs=observation()
        self.assertTrue(b.add(obs)); self.assertFalse(b.add(obs))
        with self.assertRaises(ContractError): b.add(observation(value=999))
        with self.assertRaises(ContractError): b.add(observation(scope=Scope('B','v1','session_A')))
        self.assertEqual(len(b.consume(0)),1); self.assertEqual(b.consume(0),())

    def test_bounded_and_explicit_pruning(self):
        b=ObservationBuffer(SCOPE,max_records=2)
        b.add(observation()); b.add(observation(revision=1,available=1))
        with self.assertRaises(ContractError): b.add(observation(revision=2,available=2))
        b.consume(1); self.assertEqual(b.prune_before(2),1)
        b.add(observation(revision=2,available=2)); self.assertEqual(b.visible(2)[0].header.revision,2)

class IndependentCloud(unittest.TestCase):
    def test_future_result_rejected_then_accepted_then_duplicate_rejected(self):
        reg,req=registered(); res=result(req,available=100)
        self.assertFalse(receive(reg,res,now=99).accepted)
        self.assertTrue(receive(reg,res,now=100).accepted)
        self.assertFalse(receive(reg,res,now=101).accepted)

    def test_result_predating_dispatch_is_rejected(self):
        reg,req=registered(dispatch=100)
        self.assertFalse(receive(reg,result(req,available=0),now=100).accepted)

    def test_all_three_scope_dimensions_are_checked(self):
        for scope in (Scope('B','v1','session_A'),Scope('persona_A','v2','session_A'),Scope('persona_A','v1','session_B')):
            reg,req=registered()
            with self.subTest(scope=scope): self.assertFalse(receive(reg,result(req,scope=scope)).accepted)

    def test_changed_question_claim_premise_or_expiry_rejected(self):
        cases=[dict(question=1),dict(claims=set()),dict(premises={'premise':2}),dict(now=1001)]
        for changes in cases:
            reg,req=registered(premises={'premise':1})
            kwargs={'premises':{'premise':1}}; kwargs.update(changes)
            with self.subTest(changes=changes): self.assertFalse(receive(reg,result(req),**kwargs).accepted)

    def test_state_tick_with_tracked_dependencies_is_allowed(self):
        reg,req=registered(premises={'premise':1})
        self.assertTrue(receive(reg,result(req),state=5,premises={'premise':1}).accepted)
        reg,req=registered()
        self.assertFalse(receive(reg,result(req),state=5).accepted)

    def test_cancelled_request_cannot_return(self):
        reg,req=registered(); reg.cancel_all(5)
        self.assertFalse(receive(reg,result(req),now=10).accepted)
        self.assertEqual(reg.status_view('request',10)['status'],'cancelled')

    def test_transport_success_does_not_promote_evidence(self):
        rt=SessionRuntime(SCOPE,'voice')
        claim=Claim(header(),'claim','Not yet verified','pending',(),True,1000)
        rt.add_claim(claim)
        req=AsyncRequest(header(request='request'),0,0,('claim',),{},1000)
        rt.register_request(req); rt.requests.dispatch('request',0); rt.advance_to(10)
        self.assertTrue(rt.receive_result(result(req)).accepted)
        self.assertFalse(rt.claims['claim'].can_assert(10))
        self.assertIsNone(rt.playback)

class IndependentPlayback(unittest.TestCase):
    def test_generated_queued_and_partial_played_are_distinct(self):
        p=PlaybackLedger(SCOPE,'u'); p.generate(packet(),0)
        self.assertEqual((p.generated_until_ms,p.queued_until_ms,p.played_until_ms),(80,0,0))
        with self.assertRaises(ContractError): ack(p,80,now=0)
        p.queue(0,scope=SCOPE,now_ms=0); ack(p,40,now=40)
        self.assertEqual(p.said_claim_ids,())
        ack(p,80,now=80); self.assertEqual(p.said_claim_ids,('c0',))

    def test_seeded_feasible_playback_upper_bound(self):
        for seed in range(30):
            rng=random.Random(seed); p=PlaybackLedger(SCOPE,'u',max_queue_ms=8000)
            now=0; prior_end=0; intervals=[]
            for sequence in range(20):
                now += rng.randint(0,160)
                p.generate(packet(sequence,available=now),now)
                p.queue(sequence,scope=SCOPE,now_ms=now)
                start=max(now,prior_end); prior_end=start+80; intervals.append((start,prior_end))
            feasible=sum(max(0,min(now,end)-start) for start,end in intervals)
            with self.subTest(seed=seed),self.assertRaises(ContractError): ack(p,feasible+1,now=now)
            ack(p,feasible,now=now)
            self.assertEqual(p.played_until_ms,feasible)

    def test_idle_time_does_not_credit_newly_queued_audio(self):
        p=PlaybackLedger(SCOPE,'u'); p.generate(packet(),0); p.queue(0,scope=SCOPE,now_ms=0)
        ack(p,80,now=80)
        p.generate(packet(1,available=1000),1000); p.queue(1,scope=SCOPE,now_ms=1000)
        with self.assertRaises(ContractError): ack(p,160,now=1000)
        self.assertEqual(p.played_until_ms,80)
        ack(p,160,now=1080); self.assertEqual(p.said_claim_ids,('c0','c1'))

    def test_stop_preserves_played_discards_unsaid_and_blocks_late_callback(self):
        p=PlaybackLedger(SCOPE,'u'); p.generate(packet(),0); p.generate(packet(1),0)
        p.queue(0,scope=SCOPE,now_ms=0); p.queue(1,scope=SCOPE,now_ms=0)
        ack(p,80,now=80); stop(p,100,now=100)
        self.assertEqual(p.said_claim_ids,('c0',)); self.assertEqual(p.snapshot()['active_queue_ms'],0)
        with self.assertRaises(ContractError): ack(p,160,now=200)
        with self.assertRaises(ContractError): p.queue(2,scope=SCOPE,now_ms=200)
        with self.assertRaises(ContractError): p.generate(packet(2,available=200),200)

    def test_critical_cursor_properties_cannot_be_overwritten(self):
        p=PlaybackLedger(SCOPE,'u')
        for key in ('generated_until_ms','queued_until_ms','played_until_ms'):
            with self.subTest(key=key), self.assertRaises((AttributeError,ContractError)):
                setattr(p,key,999)

    def test_callback_binds_utterance_not_just_session(self):
        p=PlaybackLedger(SCOPE,'new'); p.generate(packet(utterance='new'),0); p.queue(0,scope=SCOPE,now_ms=0)
        self.assertIn('utterance_id',inspect.signature(p.acknowledge_played).parameters,
                      'Callbacks need utterance identity to reject late callbacks from same-session old utterances')
        with self.assertRaises(ContractError): ack(p,80,now=80,utterance='old')
        self.assertEqual(p.played_until_ms,0)

    def test_queue_duration_bound_and_foreign_session(self):
        p=PlaybackLedger(SCOPE,'u',max_queue_ms=80)
        p.generate(packet(),0); p.generate(packet(1),0); p.queue(0,scope=SCOPE,now_ms=0)
        with self.assertRaises(ContractError): p.queue(1,scope=SCOPE,now_ms=0)
        with self.assertRaises(ContractError): p.acknowledge_played(80,scope=Scope('persona_A','v1','foreign'),utterance_id='u',now_ms=80)

class IndependentMemoryAndSwitch(unittest.TestCase):
    def test_memory_keeps_provenance_history_and_availability(self):
        m=SessionMemory(SCOPE)
        old=MemoryEvent(header(),'m','event','reported statement','reported'); m.put(old,0)
        new=MemoryEvent(header(time=10,revision=1,provenance=('source',)),'m','event','corrected statement','corrected')
        m.put(new,10)
        self.assertEqual(m.get('m',scope=SCOPE,now_ms=5),old)
        self.assertEqual(m.get('m',scope=SCOPE,now_ms=10),new)
        self.assertEqual(len(m.history('m',scope=SCOPE,now_ms=10)),2)
        with self.assertRaises(ContractError): m.get('m',scope=Scope('B','v1','session_A'),now_ms=10)

    def test_switch_requires_flush_and_blocks_old_scope_and_reuse(self):
        r=SessionRuntime(SCOPE,'voice')
        p=r.start_utterance('u'); p.generate(packet(),0); p.queue(0,scope=SCOPE,now_ms=0)
        r.begin_switch()
        with self.assertRaises(ContractError): r.commit_switch(Scope('B','v1','session_B'),'other')
        kwargs={'scope':SCOPE,'now_ms':80}
        if 'utterance_id' in inspect.signature(p.confirm_stopped).parameters: kwargs['utterance_id']='u'
        p.confirm_stopped(80,**kwargs); r.advance_to(80)
        old=r.commit_switch(Scope('B','v1','session_B'),'other')
        self.assertEqual(old['playback']['said_claim_ids'],['c0'])
        self.assertEqual(r.clock.now_ms,0); self.assertEqual(r.cache_handles,{})
        with self.assertRaises(ContractError): r.observations.add(observation())
        r.begin_switch()
        with self.assertRaises(ContractError): r.commit_switch(SCOPE,'voice')

    def test_switch_archive_return_is_detached(self):
        r=SessionRuntime(SCOPE,'voice'); r.begin_switch()
        old=r.commit_switch(Scope('B','v1','session_B'),'other')
        old['persona_id']='forged'; old['claim_ids'].append('forged')
        archived=r.archived_session(SCOPE.session_id)
        self.assertEqual(archived['persona_id'],SCOPE.persona_id)
        self.assertEqual(archived['claim_ids'],[])
        archived['claim_ids'].append('other_forged')
        self.assertEqual(r.archived_session(SCOPE.session_id)['claim_ids'],[])

    def test_switch_archive_retains_old_memory(self):
        r=SessionRuntime(SCOPE,'voice')
        r.memory.put(MemoryEvent(header(),'m','event','must survive switch','reported'),0)
        r.begin_switch(); old=r.commit_switch(Scope('B','v1','session_B'),'other')
        self.assertIn('must survive switch',json.dumps(old,default=str,ensure_ascii=False),
                      'Switch must return/preserve old M+ snapshot before replacing memory namespace')
        self.assertIsNone(r.memory.get('m',scope=r.scope,now_ms=0))


class IndependentValidationBypass(unittest.TestCase):
    def test_namespace_identity_cannot_change_after_construction(self):
        instances=[ObservationBuffer(SCOPE),SessionMemory(SCOPE),RequestRegistry(SCOPE),PlaybackLedger(SCOPE,'u')]
        other=Scope('persona_B','v1','session_B')
        for value in instances:
            with self.subTest(type=type(value).__name__), self.assertRaises((AttributeError,ContractError)):
                value.scope=other
        p=PlaybackLedger(SCOPE,'u')
        with self.assertRaises((AttributeError,ContractError)): p.utterance_id='old'

    def test_runtime_dependency_and_claim_mappings_are_read_only(self):
        r=SessionRuntime(SCOPE,'voice')
        with self.assertRaises(TypeError): r.premise_versions['untracked']=0
        with self.assertRaises(TypeError): r.claims['foreign']=Claim(header(scope=Scope('B','v1','s')),'foreign','text','pending',(),False,100)

    def test_default_claim_slot_budget_is_enforced(self):
        r=SessionRuntime(SCOPE,'voice')
        for i in range(8): r.add_claim(Claim(header(),f'c{i}','text','pending',(),False,100))
        with self.assertRaises(ContractError): r.add_claim(Claim(header(),'overflow','text','pending',(),False,100))

    def test_runtime_snapshot_does_not_reveal_future_component_state(self):
        r=SessionRuntime(SCOPE,'voice'); p=r.start_utterance('u')
        p.generate(packet(),0); p.queue(0,scope=SCOPE,now_ms=0)
        try:
            ack(p,80,now=80)
            snapshot=r.snapshot()
        except ContractError:
            return
        self.assertGreaterEqual(snapshot['now_ms'],80,
                                'A runtime t0 snapshot cannot already know playback confirmed at t80')

    def test_runtime_cannot_reuse_utterance_id_in_same_session(self):
        runtime=SessionRuntime(SCOPE,'voice')
        old=runtime.start_utterance('u'); stop(old,0,now=0)
        with self.assertRaises(ContractError): runtime.start_utterance('u')

    def test_json_exponent_overflow_rejected(self):
        from encephalon.serialization import strict_loads
        for text in ('1e9999','-1e9999','{"x":1e9999}','[1e9999]'):
            with self.subTest(text=text), self.assertRaises(ContractError): strict_loads(text)

    def test_malformed_record_fields_do_not_raise_unhandled_exceptions(self):
        from encephalon.serialization import record_to_dict,record_from_dict
        bases=[observation(),Claim(header(),'claim','text','pending',(),True,100),
               AsyncRequest(header(request='r'),0,0,('claim',),{},100),
               packet(),MemoryEvent(header(),'m','e','text','reported')]
        for base in bases:
            data=record_to_dict(base)
            for field in data:
                for value in (None,[],{},True,False,0,1,''):
                    candidate=dict(data); candidate[field]=value
                    with self.subTest(record=type(base).__name__,field=field,value=value):
                        try: record_from_dict(candidate)
                        except ContractError: pass
                        except Exception as exc: self.fail(f'unhandled {type(exc).__name__}: {exc}')

if __name__ == '__main__': unittest.main()
