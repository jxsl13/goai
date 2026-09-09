# frozen_string_literal: true

require 'json'
require 'minitest/autorun'
require 'open3'
require 'tmpdir'
require_relative 'analyze_sites'

class AllocationSitesTest < Minitest::Test
  CALLER = AllocationSites::CALLER
  WORKER = AllocationSites::WORKER
  SCRIPT = File.expand_path('analyze_sites.rb', __dir__)

  def counters(total, mallocs, frees, heap_alloc = 500, heap_objects = 50, num_gc = 1)
    {
      'total_alloc' => total,
      'mallocs' => mallocs,
      'frees' => frees,
      'heap_alloc' => heap_alloc,
      'heap_objects' => heap_objects,
      'num_gc' => num_gc
    }
  end

  def frame(pc, function, file = '/old/source.go', line = 10)
    {'pc' => pc, 'function' => function, 'file' => file, 'line' => line}
  end

  def pcs(*prefix)
    prefix + Array.new(32 - prefix.length, '0x0')
  end

  def row(ordinal, slot, objects, freed, pc_prefix, functions, options = {})
    frames = functions.each_with_index.map do |function, index|
      frame(options.fetch(:frame_pcs, pc_prefix)[[index, pc_prefix.length - 1].min], function,
            options.fetch(:file, '/old/source.go'), options.fetch(:line, 10) + index)
    end
    {
      'ordinal' => ordinal,
      'alloc_bytes' => slot * objects,
      'free_bytes' => slot * freed,
      'alloc_objects' => objects,
      'free_objects' => freed,
      'pcs' => pcs(*pc_prefix),
      'frames' => frames
    }
  end

  def zero_row(ordinal, pc_prefix = [], functions = [])
    row = row(ordinal, 1, 0, 0, pc_prefix, functions)
    row['alloc_bytes'] = 0
    row
  end

  def snapshot(rows)
    {'reported_count' => rows.length, 'ok' => true, 'rows' => rows}
  end

  def fixture
    big = (1 << 53) + 17
    pre_rows = [
      row(0, 16, 5, 1, %w[0x10 0x11], [CALLER, WORKER, 'repeat', 'repeat']),
      row(1, 16, 2, 0, %w[0x10 0x11], [CALLER, WORKER, 'repeat', 'repeat'], file: '/shift/a.go', line: 91),
      row(2, 16, 3, 0, %w[0x20], [CALLER, WORKER, 'repeat', 'repeat']),
      row(3, 32, 4, 1, %w[0x30], [WORKER, 'inline', 'inline']),
      row(4, 8, big, 2, %w[0x40], ['runtime.other']),
      row(5, 64, 2, 0, %w[0x50], ['unchanged']),
      zero_row(6),
      zero_row(7, %w[0x60], [''])
    ]
    post_rows = [
      row(0, 16, 8, 2, %w[0x10 0x11], [CALLER, WORKER, 'repeat', 'repeat'], file: '/new/source.go', line: 191),
      row(1, 16, 5, 0, %w[0x20], [CALLER, WORKER, 'repeat', 'repeat'], file: '/new/source.go', line: 291),
      row(2, 32, 7, 2, %w[0x30], [WORKER, 'inline', 'inline']),
      row(3, 8, big + 2, 3, %w[0x40], ['runtime.other'], file: '/new/other.go', line: 300),
      row(4, 8, 3, 1, %w[0x41], ['runtime.other'], file: '/new/other.go', line: 301),
      row(5, 64, 2, 0, %w[0x50], ['unchanged']),
      row(6, 24, 2, 0, %w[0x70], ['post.only']),
      zero_row(7),
      zero_row(8, %w[0x61], [''])
    ]
    boundaries = {
      'pre_raw_before' => counters(1000, 100, 20, 500, 50, 1),
      'start' => counters(1000, 100, 20, 490, 49, 1),
      'end' => counters(1400, 120, 25, 600, 61, 1),
      'post_gc' => counters(1450, 123, 28, 430, 42, 2),
      'post_raw' => counters(1470, 125, 29, 450, 44, 2)
    }
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
      'boundaries' => boundaries,
      'pre' => snapshot(pre_rows),
      'post' => snapshot(post_rows),
      'region_error' => '',
      'errors' => []
    }
    tail = {
      'schema' => AllocationSites::EXPECTED_TAIL_SCHEMA,
      'tail' => counters(1500, 128, 30, 440, 43, 3),
      'report_write_excluded' => true
    }
    [capture, tail]
  end

  def deep_copy(value)
    Marshal.load(Marshal.dump(value))
  end

  def assert_invalid(capture, tail, pattern = nil)
    error = assert_raises(AllocationSites::Invalid) { AllocationSites.analyze(capture, tail) }
    assert_match(pattern, error.message) if pattern
    error
  end

  def test_exact_derived_aggregation_and_independent_values
    capture, tail = fixture
    before_capture = deep_copy(capture)
    before_tail = deep_copy(tail)
    derived = AllocationSites.analyze(capture, tail)

    assert_equal before_capture, capture
    assert_equal before_tail, tail
    assert_equal %w[schema control n gomaxprocs caller_function worker_function windows raw_keys normalized_sites zero_active groups profile_totals immediate_minus_profile interpretation], derived.keys
    assert_equal [
      {'snapshot' => 'pre', 'ordinal' => 6}, {'snapshot' => 'pre', 'ordinal' => 7},
      {'snapshot' => 'post', 'ordinal' => 7}, {'snapshot' => 'post', 'ordinal' => 8}
    ], derived['zero_active']
    assert_equal 7, derived['raw_keys'].length
    assert_equal 5, derived['normalized_sites'].length

    caller = derived['groups']['caller']
    assert_equal 2, caller['raw_key_count']
    assert_equal 1, caller['normalized_site_count']
    assert_equal({'alloc_bytes' => 160, 'free_bytes' => 16, 'alloc_objects' => 10, 'free_objects' => 1}, caller['pre'])
    assert_equal({'alloc_bytes' => 208, 'free_bytes' => 32, 'alloc_objects' => 13, 'free_objects' => 2}, caller['post'])
    assert_equal({'alloc_bytes' => 48, 'free_bytes' => 16, 'alloc_objects' => 3, 'free_objects' => 1}, caller['delta'])
    caller_site = derived['normalized_sites'].find { |site| site['group'] == 'caller' }
    assert_equal 2, caller_site['raw_key_cardinality']
    assert_equal [2, 3], caller_site['raw_key_indices']

    worker = derived['groups']['worker_temporally_associated']
    assert_equal 1, worker['raw_key_count']
    assert_equal 1, worker['normalized_site_count']
    assert_equal({'alloc_bytes' => 96, 'free_bytes' => 32, 'alloc_objects' => 3, 'free_objects' => 1}, worker['delta'])
    assert_equal({'alloc_bytes' => 128, 'free_bytes' => 32, 'alloc_objects' => 4, 'free_objects' => 1}, worker['pre'])
    assert_equal({'alloc_bytes' => 224, 'free_bytes' => 64, 'alloc_objects' => 7, 'free_objects' => 2}, worker['post'])
    other = derived['groups']['other']
    assert_equal 4, other['raw_key_count']
    assert_equal 3, other['normalized_site_count']
    big = (1 << 53) + 17
    assert_equal({'alloc_bytes' => 8 * big + 128, 'free_bytes' => 16, 'alloc_objects' => big + 2, 'free_objects' => 2}, other['pre'])
    assert_equal({'alloc_bytes' => 8 * big + 216, 'free_bytes' => 32, 'alloc_objects' => big + 9, 'free_objects' => 4}, other['post'])
    assert_equal({'alloc_bytes' => 88, 'free_bytes' => 16, 'alloc_objects' => 7, 'free_objects' => 2}, other['delta'])
    assert_equal({'alloc_bytes' => 8 * big + 416, 'free_bytes' => 64, 'alloc_objects' => big + 16, 'free_objects' => 4}, derived['profile_totals']['pre'])
    assert_equal({'alloc_bytes' => 8 * big + 648, 'free_bytes' => 128, 'alloc_objects' => big + 29, 'free_objects' => 8}, derived['profile_totals']['post'])
    assert_equal({'alloc_bytes' => 232, 'free_bytes' => 64, 'alloc_objects' => 13, 'free_objects' => 4}, derived['profile_totals']['delta'])
    assert_equal({'alloc_bytes' => 168, 'alloc_objects' => 7, 'free_objects' => 1}, derived['immediate_minus_profile'])

    assert_equal counters(0, 0, 0, -10, -1, 0), derived['windows']['pre_snapshot']
    assert_equal counters(400, 20, 5, 110, 12, 0), derived['windows']['region']
    assert_equal counters(50, 3, 3, -170, -19, 1), derived['windows']['post_gc']
    assert_equal counters(20, 2, 1, 20, 2, 0), derived['windows']['post_snapshot']
    assert_equal counters(30, 3, 1, -10, -1, 1), derived['windows']['serialization']

    collision = derived['raw_keys'].find { |raw| raw['pcs'] == %w[0x10 0x11] }
    assert_equal [0, 1], collision['pre_ordinals']
    assert_equal 2, collision['pre_cardinality']
    assert_equal [CALLER, WORKER, 'repeat', 'repeat'], collision['functions']
    unchanged = derived['raw_keys'].find { |raw| raw['functions'] == ['unchanged'] }
    assert_equal({'alloc_bytes' => 0, 'free_bytes' => 0, 'alloc_objects' => 0, 'free_objects' => 0}, unchanged['delta'])
    post_only = derived['raw_keys'].find { |raw| raw['functions'] == ['post.only'] }
    assert_equal [], post_only['pre_ordinals']
    assert_equal({'alloc_bytes' => 48, 'free_bytes' => 0, 'alloc_objects' => 2, 'free_objects' => 0}, post_only['delta'])
    assert_operator derived['profile_totals']['pre']['alloc_objects'], :>, (1 << 53)

    encoded = JSON.generate(derived)
    assert_equal derived, AllocationSites.parse(encoded)
    assert_equal encoded, JSON.generate(AllocationSites.analyze(capture, tail))
    assert_equal((0...derived['raw_keys'].length).to_a, derived['raw_keys'].map { |raw| raw['index'] })
    assert_equal((0...derived['normalized_sites'].length).to_a, derived['normalized_sites'].map { |site| site['index'] })
    assert_equal derived['raw_keys'].map { |raw| raw['index'] }.sort,
                 derived['normalized_sites'].flat_map { |site| site['raw_key_indices'] }.sort
    assert_equal((0..5).to_a, derived['raw_keys'].flat_map { |raw| raw['pre_ordinals'] }.sort)
    assert_equal((0..6).to_a, derived['raw_keys'].flat_map { |raw| raw['post_ordinals'] }.sort)
    assert_equal((0..7).to_a,
                 (derived['raw_keys'].flat_map { |raw| raw['pre_ordinals'] } +
                  derived['zero_active'].select { |zero| zero['snapshot'] == 'pre' }.map { |zero| zero['ordinal'] }).sort)
    assert_equal((0..8).to_a,
                 (derived['raw_keys'].flat_map { |raw| raw['post_ordinals'] } +
                  derived['zero_active'].select { |zero| zero['snapshot'] == 'post' }.map { |zero| zero['ordinal'] }).sort)
    assert_equal %w[index slot_size pcs functions group pre post delta pre_ordinals post_ordinals pre_cardinality post_cardinality], derived['raw_keys'][0].keys
    assert_equal %w[index slot_size functions group pre post delta raw_key_indices raw_key_cardinality], derived['normalized_sites'][0].keys
    assert_equal %w[pre post delta raw_key_count normalized_site_count], derived['groups']['caller'].keys
    assert_equal %w[alloc_bytes free_bytes alloc_objects free_objects], derived['profile_totals']['pre'].keys
  end

  def test_strict_parse_rejects_duplicate_nonfinite_and_trailing_content
    assert_equal({'a' => [{'b' => 1}]}, AllocationSites.parse('{"a":[{"b":1}]}'))
    ['{"a":1,"a":2}', '{"a":{"b":1,"b":2}}', '{"a":NaN}', '{"a":Infinity}', '{"a":1} trailing'].each do |text|
      assert_raises(AllocationSites::Invalid, text) { AllocationSites.parse(text) }
    end
    assert_raises(AllocationSites::Invalid) { AllocationSites.parse(nil) }
  end

  def test_parse_rejects_overflowed_exponents_recursively_but_retains_finite_floats
    assert_equal 100.0, AllocationSites.parse('1e2')
    assert_equal({'finite' => [-0.25, {'exponent' => 6.25e12}]},
                 AllocationSites.parse('{"finite":[-0.25,{"exponent":6.25e12}]}'))

    {
      '1e999' => /\$: nonfinite JSON number/,
      '-1e999' => /\$: nonfinite JSON number/,
      '{"outer":[0,1e999]}' => /\$\["outer"\]\[1\]: nonfinite JSON number/,
      '[{"finite":1.5},{"mixed":[null,-1e999,true]}]' => /\$\[1\]\["mixed"\]\[1\]: nonfinite JSON number/
    }.each do |text, path_pattern|
      error = assert_raises(AllocationSites::Invalid, text) { AllocationSites.parse(text) }
      assert_match path_pattern, error.message
    end

    capture, tail = fixture
    parsed_capture = AllocationSites.parse(JSON.generate(capture).sub('"n":1024', '"n":1.024e3'))
    assert_equal 1024.0, parsed_capture['n']
    assert_invalid(parsed_capture, tail, /capture\.n/)
  end

  def test_exact_keys_missing_unknown_and_wrong_container_types
    capture, tail = fixture
    capture['extra'] = 1
    assert_invalid(capture, tail, /unknown keys/)
    capture, tail = fixture
    capture.delete('schema')
    assert_invalid(capture, tail, /missing keys/)
    capture, tail = fixture
    capture['boundaries']['start']['extra'] = 1
    assert_invalid(capture, tail, /unknown keys/)
    capture, tail = fixture
    capture['post']['rows'][0]['frames'][0]['extra'] = 1
    assert_invalid(capture, tail, /unknown keys/)
    capture, tail = fixture
    capture['pre']['rows'][0].delete('pcs')
    assert_invalid(capture, tail, /missing keys/)
    capture, tail = fixture
    tail['tail'] = nil
    assert_invalid(capture, tail, /must be an object/)
  end

  def test_every_capture_scalar_and_environment_guard
    replacements = {
      'schema' => 'wrong', 'control' => 'wrong', 'n' => 1023, 'warmup_calls' => 2,
      'completed_calls' => 1023, 'go_version' => '', 'goos' => 'linux', 'goarch' => 'amd64',
      'gomaxprocs' => 2, 'godebug' => 'memprofilerate=2', 'gogc' => 'off',
      'gomemlimit' => '1GiB', 'profile_rate_before' => 2, 'profile_rate_after' => 2,
      'raw_capacity' => 1, 'caller_function' => 'caller', 'worker_function' => 'worker',
      'region_error' => 'boom', 'errors' => ['bad']
    }
    replacements.each do |field, replacement|
      capture, tail = fixture
      capture[field] = replacement
      assert_invalid(capture, tail, /#{Regexp.escape(field)}/)
    end
    capture, tail = fixture
    capture['gomaxprocs'] = 1
    assert_equal 1, AllocationSites.analyze(capture, tail)['gomaxprocs']
  end

  def test_tail_scalar_guards_and_counter_ranges
    capture, tail = fixture
    tail['schema'] = 'wrong'
    assert_invalid(capture, tail, /tail.schema/)
    capture, tail = fixture
    tail['report_write_excluded'] = false
    assert_invalid(capture, tail, /report_write_excluded/)
    capture, tail = fixture
    tail['tail']['total_alloc'] = AllocationSites::U64_MAX + 1
    assert_invalid(capture, tail, /total_alloc/)
    capture, tail = fixture
    tail['tail']['num_gc'] = AllocationSites::U32_MAX + 1
    assert_invalid(capture, tail, /num_gc/)
  end

  def test_numeric_float_string_boolean_and_null_rejected
    [1.0, '1', true, nil].each do |bad|
      capture, tail = fixture
      capture['boundaries']['end']['mallocs'] = bad
      assert_invalid(capture, tail, /must be an integer/)
    end
    capture, tail = fixture
    capture['post']['rows'][0]['frames'] = nil
    assert_invalid(capture, tail, /must be an array/)
  end

  def test_snapshot_status_capacity_count_and_ordinal_guards
    capture, tail = fixture
    capture['pre']['ok'] = false
    assert_invalid(capture, tail, /pre.ok/)
    capture, tail = fixture
    capture['post']['reported_count'] = 65_537
    capture['post']['rows'] = []
    assert_invalid(capture, tail, /capacity/)
    capture, tail = fixture
    capture['pre']['reported_count'] += 1
    assert_invalid(capture, tail, /does not match/)
    capture, tail = fixture
    capture['post']['rows'][1]['ordinal'] = 7
    assert_invalid(capture, tail, /ordinal/)
  end

  def test_pc_and_frame_validation
    mutations = [
      lambda { |c| c['pre']['rows'][0]['pcs'] = c['pre']['rows'][0]['pcs'][0, 31] },
      lambda { |c| c['pre']['rows'][0]['pcs'][0] = '0X10' },
      lambda { |c| c['pre']['rows'][0]['pcs'][0] = '0x010' },
      lambda { |c| c['pre']['rows'][0]['pcs'][0] = '0x10000000000000000' },
      lambda { |c| c['pre']['rows'][0]['pcs'][1] = '0x0'; c['pre']['rows'][0]['pcs'][2] = '0x2' },
      lambda { |c| c['pre']['rows'][0]['frames'][0]['pc'] = '12' },
      lambda { |c| c['pre']['rows'][0]['frames'][0]['function'] = '' },
      lambda { |c| c['pre']['rows'][0]['frames'][0]['file'] = nil },
      lambda { |c| c['pre']['rows'][0]['frames'][0]['line'] = -1 },
      lambda { |c| c['pre']['rows'][0]['pcs'] = pcs; c['pre']['rows'][0]['frames'] = [] }
    ]
    mutations.each do |mutation|
      capture, tail = fixture
      mutation.call(capture)
      assert_invalid(capture, tail)
    end
    capture, tail = fixture
    capture['pre']['rows'][6]['frames'] = [frame('0x1', '', '', 0)]
    assert_invalid(capture, tail, /prefix is empty/)
    capture, tail = fixture
    capture['pre']['rows'][0]['frames'][0]['line'] = 1 << 100
    assert AllocationSites.analyze(capture, tail)
  end

  def test_slot_identity_and_raw_counter_guards
    mutations = [
      lambda { |r| r['alloc_objects'] = 0 },
      lambda { |r| r['alloc_bytes'] = 0 },
      lambda { |r| r['alloc_bytes'] += 1 },
      lambda { |r| r['free_objects'] = r['alloc_objects'] + 1 },
      lambda { |r| r['free_bytes'] = r['alloc_bytes'] + 1 },
      lambda { |r| r['free_bytes'] += 1 },
      lambda { |r| r['alloc_objects'] = -1 },
      lambda { |r| r['alloc_bytes'] = AllocationSites::S64_MAX + 1 }
    ]
    mutations.each do |mutation|
      capture, tail = fixture
      mutation.call(capture['pre']['rows'][0])
      assert_invalid(capture, tail)
    end
  end

  def test_decrease_is_checked_per_raw_key_before_normalization
    capture, tail = fixture
    # Both raw keys normalize together. One decreases while the other increases enough
    # that a post-normalization-only check would incorrectly accept the fixture.
    capture['post']['rows'][0] = row(0, 16, 6, 1, %w[0x10 0x11], [CALLER, WORKER, 'repeat', 'repeat'])
    capture['post']['rows'][1] = row(1, 16, 9, 1, %w[0x20], [CALLER, WORKER, 'repeat', 'repeat'])
    assert_invalid(capture, tail, /decreasing cumulative profile counter/)
  end

  def test_symbolization_mismatch_for_same_raw_key_rejected_but_location_shift_allowed
    capture, tail = fixture
    capture['post']['rows'][0]['frames'][0]['function'] = 'different'
    assert_invalid(capture, tail, /function stack differs/)
    capture, tail = fixture
    capture['post']['rows'][0]['frames'][0]['file'] = '/entirely/different.go'
    capture['post']['rows'][0]['frames'][0]['line'] = 999
    capture['post']['rows'][0]['frames'][0]['pc'] = '0xbeef'
    assert AllocationSites.analyze(capture, tail)
  end

  def test_boundary_decreases_pre_snapshot_movement_and_signed_window_overflow
    capture, tail = fixture
    capture['boundaries']['end']['mallocs'] = 99
    assert_invalid(capture, tail, /decreases/)
    capture, tail = fixture
    capture['boundaries']['start']['total_alloc'] += 1
    assert_invalid(capture, tail, /must not move/)
    capture, tail = fixture
    capture['boundaries']['pre_raw_before']['total_alloc'] = 0
    capture['boundaries']['start']['total_alloc'] = 0
    capture['boundaries']['end']['total_alloc'] = AllocationSites::U64_MAX
    capture['boundaries']['post_gc']['total_alloc'] = AllocationSites::U64_MAX
    capture['boundaries']['post_raw']['total_alloc'] = AllocationSites::U64_MAX
    tail['tail']['total_alloc'] = AllocationSites::U64_MAX
    assert_invalid(capture, tail, /overflow/)
  end

  def test_checked_raw_sum_product_and_window_upper_bound_overflows
    capture, tail = fixture
    left = capture['pre']['rows'][0]
    right = capture['pre']['rows'][1]
    left['alloc_bytes'] = AllocationSites::S64_MAX - 15
    left['alloc_objects'] = left['alloc_bytes'] / 16
    left['alloc_bytes'] = left['alloc_objects'] * 16
    right['alloc_bytes'] = 32
    right['alloc_objects'] = 2
    assert_invalid(capture, tail, /overflow/)

    capture, tail = fixture
    raw = capture['pre']['rows'][0]
    raw['alloc_objects'] = AllocationSites::S64_MAX
    raw['alloc_bytes'] = AllocationSites::S64_MAX
    raw['free_objects'] = 2
    raw['free_bytes'] = 2
    # A separate checked multiplication unit exercises the impossible-to-encode
    # product overflow path without allowing an invalid row through other guards.
    assert_raises(AllocationSites::Invalid) { AllocationSites.checked_mul(AllocationSites::S64_MAX, 2, '$test') }
    assert_invalid(capture, tail)

    capture, tail = fixture
    capture['boundaries']['start']['total_alloc'] = AllocationSites::S64_MAX
    capture['boundaries']['pre_raw_before']['total_alloc'] = AllocationSites::S64_MAX
    capture['boundaries']['end']['total_alloc'] = AllocationSites::U64_MAX
    capture['boundaries']['post_gc']['total_alloc'] = AllocationSites::U64_MAX
    capture['boundaries']['post_raw']['total_alloc'] = AllocationSites::U64_MAX
    tail['tail']['total_alloc'] = AllocationSites::U64_MAX
    assert_invalid(capture, tail, /overflow/)
  end

  def test_valid_near_uint64_memstats_have_small_exact_signed_windows
    capture, tail = fixture
    u = AllocationSites::U64_MAX
    g = AllocationSites::U32_MAX
    capture['boundaries'] = {
      'pre_raw_before' => counters(u - 100, u - 80, u - 60, u - 20, u - 30, g - 2),
      'start' => counters(u - 100, u - 80, u - 60, u - 22, u - 29, g - 2),
      'end' => counters(u - 90, u - 75, u - 58, u - 10, u - 25, g - 2),
      'post_gc' => counters(u - 85, u - 72, u - 55, u - 40, u - 35, g - 1),
      'post_raw' => counters(u - 82, u - 70, u - 54, u - 35, u - 33, g - 1)
    }
    tail['tail'] = counters(u - 80, u - 68, u - 53, u - 38, u - 34, g)
    windows = AllocationSites.analyze(capture, tail)['windows']
    assert_equal counters(0, 0, 0, -2, 1, 0), windows['pre_snapshot']
    assert_equal counters(10, 5, 2, 12, 4, 0), windows['region']
    assert_equal counters(5, 3, 3, -30, -10, 1), windows['post_gc']
    assert_equal counters(3, 2, 1, 5, 2, 0), windows['post_snapshot']
    assert_equal counters(2, 2, 1, -3, -1, 1), windows['serialization']
  end

  def test_zero_active_multiplicity_and_zero_pre_stack_becoming_positive_post
    capture, tail = fixture
    capture['pre']['rows'] << zero_row(8, %w[0x80], [''])
    capture['pre']['rows'] << zero_row(9, %w[0x80], [''])
    capture['pre']['rows'] << zero_row(10, %w[0x90], [CALLER])
    capture['pre']['reported_count'] = capture['pre']['rows'].length
    capture['post']['rows'] << row(9, 40, 2, 0, %w[0x90], [CALLER])
    capture['post']['reported_count'] = capture['post']['rows'].length

    derived = AllocationSites.analyze(capture, tail)
    assert_equal [
      {'snapshot' => 'pre', 'ordinal' => 8},
      {'snapshot' => 'pre', 'ordinal' => 9}
    ], derived['zero_active'].select { |zero| zero['snapshot'] == 'pre' && [8, 9].include?(zero['ordinal']) }
    assert_includes derived['zero_active'], {'snapshot' => 'pre', 'ordinal' => 10}
    promoted = derived['raw_keys'].find { |raw| raw['pcs'] == ['0x90'] }
    assert_equal [], promoted['pre_ordinals']
    assert_equal [9], promoted['post_ordinals']
    assert_equal({'alloc_bytes' => 80, 'free_bytes' => 0, 'alloc_objects' => 2, 'free_objects' => 0}, promoted['delta'])
  end

  def test_global_pc_file_and_line_shift_preserves_normalized_results
    capture_a, tail_a = fixture
    capture_b, tail_b = deep_copy([capture_a, tail_a])
    %w[pre post].each do |snapshot|
      capture_b[snapshot]['rows'].each do |raw|
        raw['pcs'].map! do |pc|
          pc == '0x0' ? pc : format('0x%x', pc[2..-1].to_i(16) + 0x1000)
        end
        raw['frames'].each do |item|
          item['pc'] = format('0x%x', item['pc'][2..-1].to_i(16) + 0x2000)
          item['file'] = "/shifted#{item['file']}"
          item['line'] += 10_000
        end
      end
    end
    derived_a = AllocationSites.analyze(capture_a, tail_a)
    derived_b = AllocationSites.analyze(capture_b, tail_b)
    refute_equal derived_a['raw_keys'].map { |raw| raw['pcs'] }, derived_b['raw_keys'].map { |raw| raw['pcs'] }
    assert_equal derived_a['normalized_sites'], derived_b['normalized_sites']
    assert_equal derived_a['groups'], derived_b['groups']
    assert_equal derived_a['profile_totals'], derived_b['profile_totals']
  end

  def overflow_post_rows(kind)
    half = AllocationSites::S64_MAX / 2 + 1
    functions = case kind
                when :normalized then [[CALLER], [CALLER]]
                when :group then [[CALLER, 'one'], [CALLER, 'two']]
                when :global then [[CALLER], [WORKER]]
                end
    [
      row(0, 1, half, 0, %w[0xa1], functions[0]),
      row(1, 1, half, 0, %w[0xa2], functions[1])
    ]
  end

  def assert_layer_overflow(kind, pattern)
    capture, tail = fixture
    capture['pre'] = snapshot([])
    capture['post'] = snapshot(overflow_post_rows(kind))
    assert_invalid(capture, tail, pattern)
  end

  def test_normalized_group_and_global_sum_overflows_are_distinct
    assert_layer_overflow(:normalized, /normalized_sites/)
    assert_layer_overflow(:group, /groups\.caller/)
    assert_layer_overflow(:global, /profile_totals/)
  end

  def test_signed_window_subtraction_lower_bound_overflow
    capture, tail = fixture
    u = AllocationSites::U64_MAX
    capture['boundaries']['pre_raw_before']['heap_alloc'] = u
    capture['boundaries']['start']['heap_alloc'] = u
    capture['boundaries']['end']['heap_alloc'] = 0
    capture['boundaries']['post_gc']['heap_alloc'] = 0
    capture['boundaries']['post_raw']['heap_alloc'] = 0
    tail['tail']['heap_alloc'] = 0
    assert_invalid(capture, tail, /windows\.region\.heap_alloc.*overflow/)
    assert_raises(AllocationSites::Invalid) { AllocationSites.checked_sub(AllocationSites::S64_MIN, 1, '$test.lower') }
  end

  def test_cli_success_exclusive_output_invalid_input_and_absolute_paths
    capture, tail = fixture
    Dir.mktmpdir('allocation-sites') do |dir|
      capture_path = File.join(dir, 'capture.json')
      tail_path = File.join(dir, 'tail.json')
      output_path = File.join(dir, 'derived.json')
      File.binwrite(capture_path, JSON.generate(capture))
      File.binwrite(tail_path, JSON.generate(tail))

      stdout, stderr, status = Open3.capture3(RbConfig.ruby, SCRIPT, capture_path, tail_path, output_path)
      assert status.success?, stderr
      assert_equal '', stdout
      assert_equal '', stderr
      assert_equal AllocationSites.analyze(capture, tail), AllocationSites.parse(File.binread(output_path))
      assert_equal 0o600, File.stat(output_path).mode & 0o777

      original = File.binread(output_path)
      _stdout, stderr, status = Open3.capture3(RbConfig.ruby, SCRIPT, capture_path, tail_path, output_path)
      refute status.success?
      assert_match(/exist/i, stderr)
      assert_equal original, File.binread(output_path)

      invalid_path = File.join(dir, 'invalid.json')
      absent_path = File.join(dir, 'absent.json')
      File.binwrite(invalid_path, '{"duplicate":1,"duplicate":2}')
      _stdout, stderr, status = Open3.capture3(RbConfig.ruby, SCRIPT, invalid_path, tail_path, absent_path)
      refute status.success?
      assert_match(/duplicate JSON key/, stderr)
      refute File.exist?(absent_path)

      relative_capture = File.basename(capture_path)
      _stdout, stderr, status = Open3.capture3(RbConfig.ruby, SCRIPT, relative_capture, tail_path, File.join(dir, 'relative.json'))
      refute status.success?
      assert_match(/absolute path/, stderr)

      _stdout, stderr, status = Open3.capture3(RbConfig.ruby, SCRIPT, capture_path, File.basename(tail_path), File.join(dir, 'relative-tail.json'))
      refute status.success?
      assert_match(/absolute path/, stderr)
      refute File.exist?(File.join(dir, 'relative-tail.json'))

      _stdout, stderr, status = Open3.capture3(RbConfig.ruby, SCRIPT, capture_path, tail_path, 'relative-derived.json')
      refute status.success?
      assert_match(/absolute path/, stderr)
      refute File.exist?(File.join(Dir.pwd, 'relative-derived.json'))
    end
  end
end
