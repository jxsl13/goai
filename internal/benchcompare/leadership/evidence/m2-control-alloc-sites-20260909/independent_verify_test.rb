# frozen_string_literal: true

require 'json'
require 'minitest/autorun'
require 'open3'
require 'tmpdir'
require_relative 'analyze_sites'

class IndependentAllocationSitesVerifyTest < Minitest::Test
  CALLER = AllocationSites::CALLER
  WORKER = AllocationSites::WORKER
  SCRIPT = File.expand_path('analyze_sites.rb', __dir__)

  def counters(total, mallocs, frees, heap_alloc, heap_objects, num_gc)
    {
      'total_alloc' => total, 'mallocs' => mallocs, 'frees' => frees,
      'heap_alloc' => heap_alloc, 'heap_objects' => heap_objects, 'num_gc' => num_gc
    }
  end

  def pc_array(prefix)
    prefix + Array.new(32 - prefix.length, '0x0')
  end

  def row(ordinal, slot, objects, freed, prefix, functions, location = '/a.go', line = 1)
    frames = functions.each_with_index.map do |function, index|
      {
        'pc' => prefix[[index, prefix.length - 1].min], 'function' => function,
        'file' => location, 'line' => line + index
      }
    end
    {
      'ordinal' => ordinal, 'alloc_bytes' => slot * objects,
      'free_bytes' => slot * freed, 'alloc_objects' => objects,
      'free_objects' => freed, 'pcs' => pc_array(prefix), 'frames' => frames
    }
  end

  def zero_row(ordinal, prefix = [], functions = [])
    value = row(ordinal, 1, 0, 0, prefix, functions)
    value['alloc_bytes'] = 0
    value
  end

  def snapshot(rows)
    {'reported_count' => rows.length, 'ok' => true, 'rows' => rows}
  end

  def fixture
    big = (1 << 53) + 9
    caller_stack = [CALLER, WORKER, 'inline.repeat', 'inline.repeat']
    pre = [
      row(0, 8, 4, 1, %w[0x20 0x21], caller_stack),
      row(1, 8, 2, 0, %w[0x20 0x21], caller_stack, '/collision.go', 90),
      row(2, 8, 3, 1, %w[0x30], caller_stack),
      row(3, 16, 2, 0, %w[0x40], [WORKER, 'work']),
      row(4, 4, big, 2, %w[0x80], ['other', 'repeat', 'repeat']),
      row(5, 32, 2, 1, %w[0x50], ['unchanged']),
      zero_row(6, %w[0x60], ['']),
      zero_row(7, %w[0x60], ['']),
      zero_row(8, %w[0x90], [CALLER])
    ]
    post = [
      row(0, 8, 7, 2, %w[0x20 0x21], caller_stack, '/moved.go', 190),
      row(1, 8, 5, 2, %w[0x30], caller_stack),
      row(2, 16, 4, 1, %w[0x40], [WORKER, 'work']),
      row(3, 4, big + 2, 3, %w[0x80], ['other', 'repeat', 'repeat']),
      row(4, 4, 3, 0, %w[0x81], ['other', 'repeat', 'repeat']),
      row(5, 32, 2, 1, %w[0x50], ['unchanged']),
      row(6, 40, 2, 0, %w[0x90], [CALLER]),
      zero_row(7, %w[0x60], ['']),
      zero_row(8, %w[0x60], [''])
    ]
    capture = {
      'schema' => AllocationSites::EXPECTED_SCHEMA,
      'control' => AllocationSites::CONTROL,
      'n' => 1024,
      'warmup_calls' => 1,
      'completed_calls' => 1024,
      'go_version' => 'go1.27.1',
      'goos' => 'darwin',
      'goarch' => 'arm64',
      'gomaxprocs' => 12,
      'godebug' => 'memprofilerate=1',
      'gogc' => '100',
      'gomemlimit' => 'off',
      'profile_rate_before' => 1,
      'profile_rate_after' => 1,
      'raw_capacity' => 65_536,
      'caller_function' => CALLER,
      'worker_function' => WORKER,
      'boundaries' => {
        'pre_raw_before' => counters(1000, 100, 20, 500, 50, 2),
        'start' => counters(1000, 100, 20, 480, 48, 2),
        'end' => counters(1200, 112, 24, 550, 54, 2),
        'post_gc' => counters(1230, 114, 26, 400, 39, 3),
        'post_raw' => counters(1240, 115, 27, 410, 40, 3)
      },
      'pre' => snapshot(pre),
      'post' => snapshot(post),
      'region_error' => '',
      'errors' => []
    }
    tail = {
      'schema' => AllocationSites::EXPECTED_TAIL_SCHEMA,
      'tail' => counters(1260, 117, 28, 405, 39, 4),
      'report_write_excluded' => true
    }
    [capture, tail]
  end

  def copy(value)
    Marshal.load(Marshal.dump(value))
  end

  def test_independently_derived_arithmetic_collisions_groups_and_windows
    capture, tail = fixture
    original = copy([capture, tail])
    result = AllocationSites.analyze(capture, tail)
    big = (1 << 53) + 9

    assert_equal original, [capture, tail]
    assert_equal 7, result['raw_keys'].length
    assert_equal 5, result['normalized_sites'].length
    assert_equal [
      {'snapshot' => 'pre', 'ordinal' => 6},
      {'snapshot' => 'pre', 'ordinal' => 7},
      {'snapshot' => 'pre', 'ordinal' => 8},
      {'snapshot' => 'post', 'ordinal' => 7},
      {'snapshot' => 'post', 'ordinal' => 8}
    ], result['zero_active']

    assert_equal({'alloc_bytes' => 72, 'free_bytes' => 16, 'alloc_objects' => 9, 'free_objects' => 2}, result['groups']['caller']['pre'])
    assert_equal({'alloc_bytes' => 176, 'free_bytes' => 32, 'alloc_objects' => 14, 'free_objects' => 4}, result['groups']['caller']['post'])
    assert_equal({'alloc_bytes' => 104, 'free_bytes' => 16, 'alloc_objects' => 5, 'free_objects' => 2}, result['groups']['caller']['delta'])
    assert_equal [3, 2], [result['groups']['caller']['raw_key_count'], result['groups']['caller']['normalized_site_count']]
    assert_equal({'alloc_bytes' => 32, 'free_bytes' => 0, 'alloc_objects' => 2, 'free_objects' => 0}, result['groups']['worker_temporally_associated']['pre'])
    assert_equal({'alloc_bytes' => 64, 'free_bytes' => 16, 'alloc_objects' => 4, 'free_objects' => 1}, result['groups']['worker_temporally_associated']['post'])
    assert_equal({'alloc_bytes' => 32, 'free_bytes' => 16, 'alloc_objects' => 2, 'free_objects' => 1}, result['groups']['worker_temporally_associated']['delta'])
    assert_equal [1, 1], [result['groups']['worker_temporally_associated']['raw_key_count'], result['groups']['worker_temporally_associated']['normalized_site_count']]
    assert_equal({'alloc_bytes' => 4 * big + 64, 'free_bytes' => 40, 'alloc_objects' => big + 2, 'free_objects' => 3}, result['groups']['other']['pre'])
    assert_equal({'alloc_bytes' => 4 * big + 84, 'free_bytes' => 44, 'alloc_objects' => big + 7, 'free_objects' => 4}, result['groups']['other']['post'])
    assert_equal({'alloc_bytes' => 20, 'free_bytes' => 4, 'alloc_objects' => 5, 'free_objects' => 1}, result['groups']['other']['delta'])
    assert_equal [3, 2], [result['groups']['other']['raw_key_count'], result['groups']['other']['normalized_site_count']]

    assert_equal({'alloc_bytes' => 4 * big + 168, 'free_bytes' => 56, 'alloc_objects' => big + 13, 'free_objects' => 5}, result['profile_totals']['pre'])
    assert_equal({'alloc_bytes' => 4 * big + 324, 'free_bytes' => 92, 'alloc_objects' => big + 25, 'free_objects' => 9}, result['profile_totals']['post'])
    assert_equal({'alloc_bytes' => 156, 'free_bytes' => 36, 'alloc_objects' => 12, 'free_objects' => 4}, result['profile_totals']['delta'])
    assert_equal({'alloc_bytes' => 44, 'alloc_objects' => 0, 'free_objects' => 0}, result['immediate_minus_profile'])
    assert_equal counters(0, 0, 0, -20, -2, 0), result['windows']['pre_snapshot']
    assert_equal counters(200, 12, 4, 70, 6, 0), result['windows']['region']
    assert_equal counters(30, 2, 2, -150, -15, 1), result['windows']['post_gc']
    assert_equal counters(10, 1, 1, 10, 1, 0), result['windows']['post_snapshot']
    assert_equal counters(20, 2, 1, -5, -1, 1), result['windows']['serialization']

    raw_collision = result['raw_keys'].find { |key| key['pcs'] == %w[0x20 0x21] }
    assert_equal [0, 1], raw_collision['pre_ordinals']
    assert_equal 2, raw_collision['pre_cardinality']
    assert_equal [CALLER, WORKER, 'inline.repeat', 'inline.repeat'], raw_collision['functions']
    assert_equal 'caller', raw_collision['group']
    normalized_collision = result['normalized_sites'].find { |site| site['slot_size'] == 4 }
    assert_equal 2, normalized_collision['raw_key_cardinality']
    unchanged = result['raw_keys'].find { |key| key['functions'] == ['unchanged'] }
    assert_equal({'alloc_bytes' => 0, 'free_bytes' => 0, 'alloc_objects' => 0, 'free_objects' => 0}, unchanged['delta'])
    promoted = result['raw_keys'].find { |key| key['pcs'] == ['0x90'] }
    assert_equal [], promoted['pre_ordinals']
    assert_equal [6], promoted['post_ordinals']
    assert_equal((0...7).to_a, result['normalized_sites'].flat_map { |site| site['raw_key_indices'] }.sort)
    assert_equal result, AllocationSites.parse(JSON.generate(result))
  end

  def test_cross_binary_pc_file_line_and_frame_pc_shift_only_changes_raw_identity
    capture_a, tail_a = fixture
    capture_b, tail_b = copy([capture_a, tail_a])
    %w[pre post].each do |snapshot_name|
      capture_b[snapshot_name]['rows'].each do |value|
        value['pcs'].map! { |pc| pc == '0x0' ? pc : format('0x%x', pc[2..-1].to_i(16) + 0x4000) }
        value['frames'].each do |frame|
          frame['pc'] = format('0x%x', frame['pc'][2..-1].to_i(16) + 0x8000)
          frame['file'] = "/different#{frame['file']}"
          frame['line'] += 5000
        end
      end
    end
    a = AllocationSites.analyze(capture_a, tail_a)
    b = AllocationSites.analyze(capture_b, tail_b)
    refute_equal a['raw_keys'].map { |key| key['pcs'] }, b['raw_keys'].map { |key| key['pcs'] }
    assert_equal a['normalized_sites'], b['normalized_sites']
    assert_equal a['groups'], b['groups']
    assert_equal a['profile_totals'], b['profile_totals']
  end

  def test_decrease_hidden_by_normalized_increase_is_rejected
    capture, tail = fixture
    stack = [CALLER, WORKER, 'inline.repeat', 'inline.repeat']
    capture['post']['rows'][0] = row(0, 8, 5, 1, %w[0x20 0x21], stack)
    capture['post']['rows'][1] = row(1, 8, 20, 2, %w[0x30], stack)
    error = assert_raises(AllocationSites::Invalid) { AllocationSites.analyze(capture, tail) }
    assert_match(/decreasing cumulative profile counter/, error.message)
  end

  def test_empty_worker_and_other_groups_are_retained
    capture, tail = fixture
    stack = [CALLER, 'only']
    capture['pre'] = snapshot([row(0, 8, 1, 0, %w[0xa], stack)])
    capture['post'] = snapshot([row(0, 8, 3, 1, %w[0xa], stack)])
    result = AllocationSites.analyze(capture, tail)
    zero = {'alloc_bytes' => 0, 'free_bytes' => 0, 'alloc_objects' => 0, 'free_objects' => 0}
    %w[worker_temporally_associated other].each do |group|
      assert_equal zero, result['groups'][group]['pre']
      assert_equal zero, result['groups'][group]['post']
      assert_equal zero, result['groups'][group]['delta']
      assert_equal 0, result['groups'][group]['raw_key_count']
      assert_equal 0, result['groups'][group]['normalized_site_count']
    end
  end

  def test_independent_range_type_key_status_and_stack_guards
    mutations = [
      lambda { |capture, _tail| capture['post']['ok'] = false },
      lambda { |capture, _tail| capture['pre']['reported_count'] = 65_537; capture['pre']['rows'] = [] },
      lambda { |capture, _tail| capture['boundaries']['end']['num_gc'] = AllocationSites::U32_MAX + 1 },
      lambda { |capture, _tail| capture['post']['rows'][0]['alloc_objects'] = 1.0 },
      lambda { |capture, _tail| capture['post']['rows'][0]['free_bytes'] += 1 },
      lambda { |capture, _tail| capture['post']['rows'][0]['pcs'][1] = '0x0'; capture['post']['rows'][0]['pcs'][2] = '0x1' },
      lambda { |capture, _tail| capture['post']['rows'][0]['frames'][0]['function'] = '' },
      lambda { |capture, _tail| capture['post']['rows'][0]['frames'][0]['unknown'] = 1 },
      lambda { |capture, _tail| capture['boundaries']['start'].delete('mallocs') },
      lambda { |_capture, tail| tail['report_write_excluded'] = nil }
    ]
    mutations.each do |mutation|
      capture, tail = fixture
      mutation.call(capture, tail)
      assert_raises(AllocationSites::Invalid) { AllocationSites.analyze(capture, tail) }
    end
    assert_raises(AllocationSites::Invalid) { AllocationSites.checked_add(AllocationSites::S64_MAX, 1, '$verify') }
    assert_raises(AllocationSites::Invalid) { AllocationSites.checked_sub(AllocationSites::S64_MIN, 1, '$verify') }
    assert_raises(AllocationSites::Invalid) { AllocationSites.checked_mul(AllocationSites::S64_MAX, 2, '$verify') }
  end

  def test_cli_is_exclusive_silent_and_invalid_input_creates_nothing
    capture, tail = fixture
    Dir.mktmpdir('independent-allocation-sites') do |dir|
      capture_path = File.join(dir, 'capture.json')
      tail_path = File.join(dir, 'tail.json')
      output_path = File.join(dir, 'derived.json')
      invalid_path = File.join(dir, 'invalid.json')
      absent_path = File.join(dir, 'absent.json')
      File.binwrite(capture_path, JSON.generate(capture))
      File.binwrite(tail_path, JSON.generate(tail))
      File.binwrite(invalid_path, '{"x":1,"x":2}')

      stdout, stderr, status = Open3.capture3(RbConfig.ruby, SCRIPT, capture_path, tail_path, output_path)
      assert status.success?
      assert_equal '', stdout
      assert_equal '', stderr
      assert_equal 0o600, File.stat(output_path).mode & 0o777
      original = File.binread(output_path)
      _stdout, _stderr, status = Open3.capture3(RbConfig.ruby, SCRIPT, capture_path, tail_path, output_path)
      refute status.success?
      assert_equal original, File.binread(output_path)
      _stdout, stderr, status = Open3.capture3(RbConfig.ruby, SCRIPT, invalid_path, tail_path, absent_path)
      refute status.success?
      assert_match(/duplicate JSON key/, stderr)
      refute File.exist?(absent_path)
    end
  end

  def test_parse_rejects_nonfinite_result_from_finite_json_exponent
    assert_raises(AllocationSites::Invalid) { AllocationSites.parse('{"overflow":1e999}') }
    assert_raises(AllocationSites::Invalid) { AllocationSites.parse('{"overflow":-1e999}') }
  end
end
