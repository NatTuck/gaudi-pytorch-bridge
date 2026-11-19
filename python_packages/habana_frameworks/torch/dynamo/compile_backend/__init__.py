###############################################################################
# Copyright (c) 2021-2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################


import habana_frameworks.torch.internal.bridge_config as bc

from . import backends

# TRACE GATE (docs/PROTOCOL.md 13): no tracing without a cache consult.
#
# Installed here because this package must be imported for the HPU dynamo backend
# to exist at all, so it precedes every compile that uses it. The gate raises
# BaseException (not Exception) so dynamo cannot swallow it into a silent eager
# fallback.
#
# The log line is not decoration: it is the only way to tell "the gate is armed"
# from "the gate never loaded", and believing you have enforcement when you do
# not is worse than having none (PROTOCOL 11).
from .trace_gate import TraceGate as _TraceGate

_TraceGate.install()

import logging as _logging

_logging.getLogger(__name__).info(
    "trace gate: installed=%s -- no trace without a cache consult (PROTOCOL 13)",
    _TraceGate.installed())

if bc.get_pt_hpu_enable_compiled_autograd():
    from .experimental import enable_compiled_autograd

    enable_compiled_autograd()
